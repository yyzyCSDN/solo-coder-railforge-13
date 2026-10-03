from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Mapping, Sequence

from railforge.core.versioning import VersionConflict, Versioned
from railforge.wagon.lease_reposition import (
    LeaseContract,
    LeaseDemand,
    LeasableWagon,
    RepositionRoute,
    optimize,
)


@dataclass(frozen=True)
class BatchCommitment:
    operation_id: str
    version: int
    selections: tuple[Any, ...]
    rejected: tuple[Any, ...]
    wagon_reserved: Mapping[str, float]
    route_reserved: Mapping[str, float]
    contract_reserved: Mapping[str, float]
    before_wagon_reserved: Mapping[str, float]
    before_route_reserved: Mapping[str, float]
    before_contract_reserved: Mapping[str, float]
    totals: Mapping[str, float]
    cost_trace: Mapping[str, Any]
    rolled_back: bool = False


@dataclass(frozen=True)
class ReservationLedger:
    version: int
    wagon_capacity: Mapping[str, float]
    route_capacity: Mapping[str, float]
    contract_capacity: Mapping[str, float]
    wagon_reserved: Mapping[str, float]
    route_reserved: Mapping[str, float]
    contract_reserved: Mapping[str, float]
    commitments: tuple[BatchCommitment, ...]


def _capacity(row: Any) -> float:
    return float(getattr(row, "quantity", getattr(row, "capacity", 0.0)))


def _reserved(row: Any) -> float:
    return float(getattr(row, "reserved", 0.0))


class LeaseReservationStore:
    """Thread-safe optimistic ledger for atomic wagon/route batch reservations."""

    def __init__(self, wagons: Sequence[Any] = (), routes: Sequence[Any] = (),
                 contracts: Sequence[Any] = ()):
        self._lock = RLock()
        self._wagon_capacity = {w.wagon_id: _capacity(w) for w in wagons}
        self._route_capacity = {r.route_id: _capacity(r) for r in routes}
        self._contract_capacity = {c.contract_id: _capacity(c) for c in contracts}
        self._wagon_reserved = {w.wagon_id: _reserved(w) for w in wagons if _reserved(w)}
        self._route_reserved = {r.route_id: _reserved(r) for r in routes if _reserved(r)}
        self._contract_reserved = {c.contract_id: _reserved(c) for c in contracts if _reserved(c)}
        self._commitments: dict[str, BatchCommitment] = {}
        self._version = 0

    def snapshot(self) -> Versioned[ReservationLedger]:
        with self._lock:
            return Versioned(self._ledger(), self._version)

    def _ledger(self) -> ReservationLedger:
        return ReservationLedger(
            self._version, dict(self._wagon_capacity), dict(self._route_capacity),
            dict(self._contract_capacity), dict(self._wagon_reserved), dict(self._route_reserved),
            dict(self._contract_reserved), tuple(self._commitments.values()),
        )

    def get(self, operation_id: str) -> BatchCommitment | None:
        with self._lock:
            return self._commitments.get(operation_id)

    def sync_catalog(self, wagons: Sequence[Any] = (), routes: Sequence[Any] = (),
                     contracts: Sequence[Any] = ()) -> None:
        """Refresh catalog capacities while retaining durable reservation totals."""
        with self._lock:
            for wagon in wagons:
                self._wagon_capacity[wagon.wagon_id] = _capacity(wagon)
                self._wagon_reserved[wagon.wagon_id] = max(_reserved(wagon),
                                                           self._wagon_reserved.get(wagon.wagon_id, 0.0))
            for route in routes:
                self._route_capacity[route.route_id] = _capacity(route)
                self._route_reserved[route.route_id] = max(_reserved(route),
                                                           self._route_reserved.get(route.route_id, 0.0))
            for contract in contracts:
                self._contract_capacity[contract.contract_id] = _capacity(contract)
                self._contract_reserved[contract.contract_id] = max(
                    _reserved(contract), self._contract_reserved.get(contract.contract_id, 0.0))

    def reserve(self, operation_id: str, plan: Any, expected_version: int | None = None) -> BatchCommitment:
        with self._lock:
            previous = self._commitments.get(operation_id)
            if previous is not None:
                return previous
            if expected_version is not None and expected_version != self._version:
                raise VersionConflict(f"expected={expected_version} actual={self._version}")
            if getattr(plan, "rejected", ()):
                rejected = [(r.request_id, r.quantity, r.reason) for r in plan.rejected]
                raise ValueError("batch reservation rejected: " + repr(rejected))

            # Work on copies so every validation failure is a rollback.
            wagon_reserved = dict(self._wagon_reserved)
            route_reserved = dict(self._route_reserved)
            contract_reserved = dict(self._contract_reserved)
            before_wagon_reserved = dict(wagon_reserved)
            before_route_reserved = dict(route_reserved)
            before_contract_reserved = dict(contract_reserved)

            for selected in plan.selections:
                wagon_capacity = self._wagon_capacity.get(selected.wagon_id)
                if wagon_capacity is None:
                    raise ValueError("unknown wagon " + selected.wagon_id)
                wagon_next = wagon_reserved.get(selected.wagon_id, 0.0) + selected.quantity
                if wagon_next > wagon_capacity + 1e-9:
                    raise ValueError("wagon capacity exhausted: " + selected.wagon_id)
                wagon_reserved[selected.wagon_id] = wagon_next

                if selected.contract_id is not None:
                    contract_capacity = self._contract_capacity.get(selected.contract_id)
                    if contract_capacity is None:
                        raise ValueError("unknown contract " + selected.contract_id)
                    contract_next = contract_reserved.get(selected.contract_id, 0.0) + selected.quantity
                    if contract_next > contract_capacity + 1e-9:
                        raise ValueError("contract capacity exhausted: " + selected.contract_id)
                    contract_reserved[selected.contract_id] = contract_next

                route_ids = getattr(selected, "route_ids", None)
                if route_ids is None and selected.route_id is not None:
                    route_ids = (selected.route_id,)
                for route_id in route_ids or ():
                    route_capacity = self._route_capacity.get(route_id)
                    if route_capacity is None:
                        raise ValueError("unknown route " + route_id)
                    route_next = route_reserved.get(route_id, 0.0) + selected.quantity
                    if route_next > route_capacity + 1e-9:
                        raise ValueError("route capacity exhausted: " + route_id)
                    route_reserved[route_id] = route_next

            cost_trace = {
                "requests": {
                    trace.request_id: {
                        "stages": [(s.stage, s.wagons, s.routes, s.options) for s in trace.stages],
                        "considered": [
                            {"wagon_id": a.wagon_id, "route_id": a.route_id, "feasible": a.feasible,
                             "reason": a.reason, "per_wagon": a.per_wagon,
                             "duration_hours": a.duration_hours, "risk": a.risk}
                            for a in trace.considered
                        ],
                        "eliminated": [
                            {"wagon_id": a.wagon_id, "route_id": a.route_id, "feasible": a.feasible,
                             "reason": a.reason, "per_wagon": a.per_wagon,
                             "duration_hours": a.duration_hours, "risk": a.risk}
                            for a in getattr(trace, "eliminated", ())
                        ],
                        "rejected_reason": trace.rejected_reason,
                    }
                    for trace in getattr(plan, "traces", ())
                },
                "selected": [
                    {"request_id": s.request_id, "wagon_id": s.wagon_id,
                     "contract_id": s.contract_id, "route_id": s.route_id,
                     "route_ids": list(getattr(s, "route_ids", (s.route_id,) if s.route_id else ())),
                     "quantity": s.quantity, "marginal_per_wagon": s.marginal_per_wagon,
                     "cost": s.cost.__dict__, "duration_hours": s.duration_hours, "risk": s.risk}
                    for s in plan.selections
                ],
                "totals": dict(plan.totals),
            }
            commitment = BatchCommitment(
                operation_id, self._version + 1, tuple(plan.selections), tuple(plan.rejected),
                wagon_reserved, route_reserved, contract_reserved,
                before_wagon_reserved, before_route_reserved, before_contract_reserved,
                dict(plan.totals), cost_trace,
            )
            self._wagon_reserved = wagon_reserved
            self._route_reserved = route_reserved
            self._contract_reserved = contract_reserved
            self._commitments[operation_id] = commitment
            self._version += 1
            return commitment

    def overloaded(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            wagons = tuple(("wagon", wid) for wid, cap in self._wagon_capacity.items()
                           if self._wagon_reserved.get(wid, 0.0) > cap + 1e-9)
            routes = tuple(("route", rid) for rid, cap in self._route_capacity.items()
                           if self._route_reserved.get(rid, 0.0) > cap + 1e-9)
            contracts = tuple(("contract", cid) for cid, cap in self._contract_capacity.items()
                              if self._contract_reserved.get(cid, 0.0) > cap + 1e-9)
            return wagons + routes + contracts

    def rollback_commitment(self, operation_id: str) -> BatchCommitment | None:
        """Compensate a domain reservation when publication after it fails."""
        with self._lock:
            commitment = self._commitments.pop(operation_id, None)
            if commitment is None:
                return
            self._wagon_reserved = dict(commitment.before_wagon_reserved)
            self._route_reserved = dict(commitment.before_route_reserved)
            self._contract_reserved = dict(commitment.before_contract_reserved)
            self._version = max(0, self._version - 1)
            return replace(commitment, rolled_back=True)

    compare_and_swap = reserve


def commit_batch(store: LeaseReservationStore, operation_id: str,
                 wagons: Sequence[LeasableWagon], contracts: Sequence[LeaseContract],
                 routes: Sequence[RepositionRoute], demands: Sequence[LeaseDemand],
                 loco_units: Sequence[Any] | None = None, turn_minutes: int = 0,
                 risk_penalty_rate: float = 1.0, expected_version: int | None = None) -> BatchCommitment:
    """Optimize one batch and reserve all selected wagons, contracts and routes atomically."""
    store.sync_catalog(wagons, routes, contracts)
    snapshot = store.snapshot().value
    effective_wagons = [
        replace(w, reserved=max(w.reserved, snapshot.wagon_reserved.get(w.wagon_id, 0.0)))
        for w in wagons
    ]
    effective_routes = [
        replace(r, reserved=max(r.reserved, snapshot.route_reserved.get(r.route_id, 0.0)))
        for r in routes
    ]
    effective_contracts = [
        replace(c, reserved=max(getattr(c, "reserved", 0.0),
                                snapshot.contract_reserved.get(c.contract_id, 0.0)))
        for c in contracts
    ]
    plan = optimize(effective_wagons, effective_contracts, effective_routes, demands,
                    loco_units, turn_minutes, risk_penalty_rate)
    return store.reserve(operation_id, plan, expected_version)


def reserve_batch(store: LeaseReservationStore, operation_id: str, plan: Any,
                  expected_version: int | None = None) -> BatchCommitment:
    return store.compare_and_swap(operation_id, plan, expected_version)
