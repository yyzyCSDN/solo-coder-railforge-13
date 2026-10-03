from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Mapping

from railforge.planning.loco_circulation import Leg, Unit, circulate


@dataclass(frozen=True)
class LeaseContract:
    contract_id: str
    wagon_type: str
    location: str
    total_quantity: int
    valid_from: date
    valid_to: date
    daily_rate: float
    maintenance_due: date
    leased_quantity: int = 0
    allowed_destinations: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LeaseDemand:
    demand_id: str
    wagon_type: str
    location: str
    on: date
    count: int
    required_days: int
    priority: int = 0


@dataclass(frozen=True)
class RepositionRoute:
    route_id: str
    origin: str
    destination: str
    capacity: int
    cost_per_wagon: float
    duration_days: float
    risk: float


@dataclass(frozen=True)
class CostBreakdown:
    lease: float
    reposition: float
    risk: float
    loco: float

    @property
    def total(self) -> float:
        return self.lease + self.reposition + self.risk + self.loco


@dataclass(frozen=True)
class CandidateTrace:
    demand_id: str
    contract_id: str
    route_id: str | None
    stage: str
    outcome: str
    reason: str
    cost: float | None = None


@dataclass(frozen=True)
class LeaseAssignment:
    demand_id: str
    contract_id: str
    count: int
    route_id: str | None
    legs: tuple[str, ...]
    units: tuple[str, ...]
    cost: CostBreakdown


@dataclass(frozen=True)
class LeasePlan:
    assignments: tuple[LeaseAssignment, ...]
    trace: tuple[CandidateTrace, ...]
    total_cost: float
    contract_updates: Mapping[str, LeaseContract]
    unit_updates: Mapping[str, Unit]


class LeasePlanError(ValueError):
    """No feasible plan for a demand; carries the candidate trace for diagnosis."""

    def __init__(self, demand_id: str, trace):
        self.demand_id = demand_id
        self.trace = tuple(trace)
        super().__init__(f"no feasible lease plan for demand {demand_id}")


def contract_key(contract_id: str) -> str:
    return f"lease-contract:{contract_id}"


def unit_key(unit_id: str) -> str:
    return f"loco-unit:{unit_id}"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_contract(contract: LeaseContract):
    if contract.total_quantity < 0 or not 0 <= contract.leased_quantity <= contract.total_quantity:
        raise ValueError("contract quantity")
    if contract.daily_rate < 0:
        raise ValueError("contract rate")
    if contract.valid_from > contract.valid_to:
        raise ValueError("contract window")


def _validate_demand(demand: LeaseDemand):
    if demand.count <= 0 or demand.required_days < 0:
        raise ValueError("demand")


def _validate_route(route: RepositionRoute):
    if route.capacity <= 0 or route.duration_days <= 0:
        raise ValueError("route")
    if route.cost_per_wagon < 0 or route.risk < 0:
        raise ValueError("route cost")


def _contract_rejection(contract: LeaseContract, demand: LeaseDemand, service_end: date) -> str | None:
    if contract.wagon_type != demand.wagon_type:
        return "wagon type mismatch"
    if contract.valid_from > demand.on or contract.valid_to < service_end:
        return "contract window does not cover service period"
    if contract.leased_quantity + demand.count > contract.total_quantity:
        return "contract quantity exhausted"
    if contract.allowed_destinations and demand.location not in contract.allowed_destinations:
        return "destination not allowed"
    return None


def _cost(contract: LeaseContract, demand: LeaseDemand, route: RepositionRoute | None,
          leg: Leg | None, risk_weight: float, loco_hourly_cost: float) -> CostBreakdown:
    lease = demand.count * contract.daily_rate * demand.required_days
    reposition = route.cost_per_wagon * demand.count if route else 0.0
    risk = route.risk * risk_weight if route else 0.0
    loco = 0.0
    if leg is not None:
        hours = (leg.arrive - leg.depart).total_seconds() / 3600.0
        loco = hours * loco_hourly_cost * leg.required_units
    return CostBreakdown(lease, reposition, risk, loco)


def optimize(contracts: list[LeaseContract], demands: list[LeaseDemand], routes: list[RepositionRoute],
             units: list[Unit], move_depart: datetime, turn_minutes: int = 0,
             risk_weight: float = 1.0, loco_hourly_cost: float = 0.0) -> LeasePlan:
    """Multi-stage lease and empty-reposition selection.

    Every demand walks contract, maintenance, location, route and locomotive
    stages; rejected candidates are traced with stage and reason, feasible
    survivors carry their cost breakdown, so the trade-offs behind the chosen
    plan stay auditable. Empty moves are scheduled just in time: each leg
    arrives by the end of the demand's need date and departs as late as the
    route duration allows, which lets one locomotive cycle through the legs
    of consecutive demands.
    """
    if turn_minutes < 0 or risk_weight < 0 or loco_hourly_cost < 0:
        raise ValueError("weights")
    depart = _utc(move_depart)
    contract_state: dict[str, LeaseContract] = {}
    for contract in contracts:
        _validate_contract(contract)
        if contract.contract_id in contract_state:
            raise ValueError("duplicate contract")
        contract_state[contract.contract_id] = contract
    route_index: dict[tuple[str, str], list[RepositionRoute]] = {}
    seen_routes: set[str] = set()
    for route in routes:
        _validate_route(route)
        if route.route_id in seen_routes:
            raise ValueError("duplicate route")
        seen_routes.add(route.route_id)
        route_index.setdefault((route.origin, route.destination), []).append(route)
    loco_state = {u.unit_id: u for u in units}
    if len(loco_state) != len(units):
        raise ValueError("duplicate unit")
    initial_contracts = dict(contract_state)
    initial_units = dict(loco_state)

    assignments = []
    trace = []
    total_cost = 0.0
    for demand in sorted(demands, key=lambda d: (-d.priority, d.on, d.demand_id)):
        _validate_demand(demand)
        service_end = demand.on + timedelta(days=demand.required_days)
        arrive_by = datetime.combine(demand.on, time.min, tzinfo=timezone.utc) + timedelta(days=1)
        candidates = []
        for contract in sorted(contract_state.values(), key=lambda c: c.contract_id):
            reason = _contract_rejection(contract, demand, service_end)
            if reason is not None:
                trace.append(CandidateTrace(demand.demand_id, contract.contract_id, None,
                                            "contract", "rejected", reason))
                continue
            if contract.maintenance_due <= service_end:
                trace.append(CandidateTrace(demand.demand_id, contract.contract_id, None,
                                            "maintenance", "rejected",
                                            f"maintenance due {contract.maintenance_due.isoformat()} "
                                            f"on or before service end {service_end.isoformat()}"))
                continue
            if contract.location == demand.location:
                route_options = [None]
            else:
                pair = route_index.get((contract.location, demand.location), [])
                if not pair:
                    trace.append(CandidateTrace(demand.demand_id, contract.contract_id, None,
                                                "location", "rejected",
                                                f"no reposition path from {contract.location} "
                                                f"to {demand.location}"))
                    continue
                route_options = []
                for route in sorted(pair, key=lambda r: r.route_id):
                    if route.capacity < demand.count:
                        trace.append(CandidateTrace(demand.demand_id, contract.contract_id, route.route_id,
                                                    "route", "rejected", "route capacity below demand"))
                        continue
                    if arrive_by - timedelta(days=route.duration_days) < depart:
                        trace.append(CandidateTrace(demand.demand_id, contract.contract_id, route.route_id,
                                                    "route", "rejected",
                                                    "no time to reposition before need date"))
                        continue
                    route_options.append(route)
                if not route_options:
                    continue
            for route in route_options:
                leg = None
                leg_units: tuple[str, ...] = ()
                trial_state = loco_state
                if route is not None:
                    leg = Leg(f"{demand.demand_id}:{route.route_id}", route.origin, route.destination,
                              arrive_by - timedelta(days=route.duration_days), arrive_by, 1)
                    try:
                        leg_result, raw_state = circulate([leg], list(loco_state.values()), turn_minutes)
                    except ValueError:
                        trace.append(CandidateTrace(demand.demand_id, contract.contract_id, route.route_id,
                                                    "loco", "rejected", "locomotive cycle infeasible"))
                        continue
                    trial_state = {uid: Unit(uid, loc, avail, due)
                                   for uid, (loc, avail, due) in raw_state.items()}
                    leg_units = tuple(leg_result[leg.leg_id])
                cost = _cost(contract, demand, route, leg, risk_weight, loco_hourly_cost)
                candidates.append((cost.total, contract.contract_id, route.route_id if route else "",
                                   contract, route, leg, trial_state, leg_units, cost))
        if not candidates:
            raise LeasePlanError(demand.demand_id, trace)
        candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        chosen_total, _, _, contract, route, leg, trial_state, leg_units, cost = candidates[0]
        for entry in candidates[1:]:
            trace.append(CandidateTrace(demand.demand_id, entry[1], entry[2] or None, "cost", "rejected",
                                        f"cost {entry[0]} exceeds chosen {chosen_total}", entry[0]))
        contract_state[contract.contract_id] = replace(
            contract, leased_quantity=contract.leased_quantity + demand.count)
        loco_state = trial_state
        total_cost += cost.total
        assignments.append(LeaseAssignment(demand.demand_id, contract.contract_id, demand.count,
                                           route.route_id if route else None,
                                           (leg.leg_id,) if leg else (), leg_units, cost))
        trace.append(CandidateTrace(demand.demand_id, contract.contract_id,
                                    route.route_id if route else None,
                                    "cost", "chosen", "minimum total cost", cost.total))
    contract_updates = {cid: c for cid, c in contract_state.items() if c != initial_contracts[cid]}
    unit_updates = {uid: u for uid, u in loco_state.items() if u != initial_units[uid]}
    return LeasePlan(tuple(assignments), tuple(trace), total_cost, contract_updates, unit_updates)
