from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence
import heapq

from railforge.planning.loco_circulation import Leg, Unit, circulate


@dataclass(frozen=True)
class LeaseContract:
    contract_id: str
    lessor: str
    origin: str
    wagon_type: str
    quantity: float
    available_on: date
    expires_on: date | None = None
    unit_rate: float = 0.0
    max_lease_days: int | None = None
    reserved: float = 0.0
    restrictions: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LeasableWagon:
    wagon_id: str
    contract_id: str | None
    wagon_type: str
    location: str
    maintenance_due: date
    quantity: float = 1.0
    reserved: float = 0.0
    available_on: date | None = None
    restrictions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RepositionRoute:
    route_id: str
    origin: str
    destination: str
    depart: Any
    arrive: Any
    capacity: float
    leg_id: str
    unit_move_cost: float = 0.0
    fixed_cost: float = 0.0
    loco_cost: float = 0.0
    risk: float = 0.0
    duration_hours: float = 0.0
    reserved: float = 0.0
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LeaseDemand:
    request_id: str
    wagon_type: str
    destination: str
    quantity: float
    ready_on: date
    lease_days: int = 0
    priority: int = 0
    forbidden_restrictions: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()
    max_cost_per_wagon: float | None = None
    max_duration_hours: float | None = None
    max_risk: float | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True)
class CostBreakdown:
    lease: float
    reposition: float
    locomotive: float
    risk_premium: float
    total: float
    per_wagon: float


@dataclass(frozen=True)
class SelectedOption:
    request_id: str
    wagon_id: str
    contract_id: str | None
    route_id: str | None
    route_ids: tuple[str, ...]
    quantity: float
    ready_on: date
    duration_hours: float
    risk: float
    marginal_per_wagon: float
    cost: CostBreakdown


@dataclass(frozen=True)
class StageCount:
    stage: str
    wagons: int
    routes: int
    options: int


@dataclass(frozen=True)
class Alternative:
    wagon_id: str
    route_id: str | None
    feasible: bool
    reason: str
    per_wagon: float
    duration_hours: float
    risk: float


@dataclass(frozen=True)
class DemandTrace:
    request_id: str
    stages: tuple[StageCount, ...]
    considered: tuple[Alternative, ...]
    eliminated: tuple[Alternative, ...] = ()
    rejected_reason: str | None = None


@dataclass(frozen=True)
class RejectedDemand:
    request_id: str
    quantity: float
    reason: str


@dataclass(frozen=True)
class LeaseRepositionPlan:
    selections: tuple[SelectedOption, ...]
    traces: tuple[DemandTrace, ...]
    rejected: tuple[RejectedDemand, ...]
    route_usage: Mapping[str, float]
    contract_usage: Mapping[str, float]
    wagon_usage: Mapping[str, float]
    cycle: Any
    totals: Mapping[str, float]


def _as_date(value: date | Any) -> date:
    return value.date() if hasattr(value, "date") else value


def _route_duration(route: RepositionRoute) -> float:
    return route.duration_hours or (route.arrive - route.depart).total_seconds() / 3600


def _lease_active(contract: LeaseContract, service_start: date, service_end: date, lease_days: int) -> bool:
    if contract.available_on > service_start:
        return False
    if contract.expires_on is not None and contract.expires_on <= service_end:
        return False
    if contract.max_lease_days is not None and lease_days > contract.max_lease_days:
        return False
    return True


def _available_on(wagon: LeasableWagon, contract: LeaseContract | None, ready_on: date) -> date:
    return wagon.available_on or (contract.available_on if contract is not None else ready_on)


def _wagon_allowed(wagon: LeasableWagon, contract: LeaseContract | None, demand: LeaseDemand,
                   service_end: date) -> tuple[bool, str | None]:
    if _available_on(wagon, contract, demand.ready_on) > demand.ready_on:
        return False, "wagon not available"
    if wagon.contract_id and contract is None:
        return False, "unknown lease contract"
    if contract is not None and (contract.wagon_type != wagon.wagon_type or contract.origin != wagon.location):
        return False, "lease contract mismatch"
    if contract is not None and not _lease_active(contract, demand.ready_on, service_end, demand.lease_days):
        return False, "lease contract window or term"
    if wagon.restrictions.intersection(demand.forbidden_restrictions):
        return False, "forbidden wagon restriction"
    if contract is not None and contract.restrictions.intersection(demand.forbidden_restrictions):
        return False, "forbidden contract restriction"
    if contract is not None and not demand.required_capabilities.issubset(contract.capabilities):
        return False, "contract capability"
    if wagon.maintenance_due <= service_end:
        return False, "maintenance due before return"
    return True, None


def _cost(contract: LeaseContract | None, path: Sequence[RepositionRoute], quantity: float,
          lease_days: int, risk_penalty_rate: float, fixed_cost: float = 0.0) -> CostBreakdown:
    lease = (contract.unit_rate if contract else 0.0) * lease_days * quantity
    move = sum(route.unit_move_cost for route in path) * quantity + fixed_cost
    locomotive = sum(route.loco_cost for route in path) * quantity
    risk = max((route.risk for route in path), default=0.0)
    risk_premium = (move + locomotive) * risk * risk_penalty_rate
    total = lease + move + locomotive + risk_premium
    return CostBreakdown(lease, move, locomotive, risk_premium, total,
                         total / quantity if quantity else 0.0)


def _option(request_id: str, wagon: LeasableWagon, contract: LeaseContract | None,
            path: Sequence[RepositionRoute], quantity: float, demand: LeaseDemand,
            risk_penalty_rate: float, fixed_cost: float = 0.0) -> SelectedOption:
    duration = sum(_route_duration(route) for route in path)
    risk = max((route.risk for route in path), default=0.0)
    last = path[-1] if path else None
    return SelectedOption(
        request_id, wagon.wagon_id, wagon.contract_id, last.route_id if last else None,
        tuple(route.route_id for route in path), quantity, demand.ready_on, duration, risk,
        _cost(contract, path, quantity, demand.lease_days, risk_penalty_rate, 0.0).per_wagon,
        _cost(contract, path, quantity, demand.lease_days, risk_penalty_rate, fixed_cost),
    )


def _loco_ok_with_turn(legs, loco_units: Sequence[Unit], turn_minutes: int) -> bool:
    try:
        circulate(list(legs), list(loco_units), turn_minutes)
    except ValueError:
        return False
    return True


def _build_path(origin: str, destination: str, available_routes: Sequence[RepositionRoute],
                demand: LeaseDemand, loco_units: Sequence[Unit], turn_minutes: int,
                check_loco: bool, route_usage: Mapping[str, float],
                used_legs: Mapping[str, Leg], earliest_on: date
                ) -> tuple[tuple[RepositionRoute, ...], dict[str, Leg]] | None:
    """Build the cheapest time-ordered empty-return path over one or more legs."""
    counter = 0
    start = (0.0, 0.0, earliest_on, counter, origin, (), {}, frozenset({origin}))
    pq = [start]

    while pq:
        move_cost, duration, arrived_on, _, node, path, path_legs, visited = heapq.heappop(pq)
        if node == destination and path:
            return path, path_legs
        arrival = None if not path else path[-1].arrive
        for route in available_routes:
            if route.origin != node or route.destination in visited:
                continue
            depart_on = _as_date(route.depart)
            if arrival is None:
                if depart_on < earliest_on:
                    continue
            elif route.depart < arrival:
                continue
            arrive_on = _as_date(route.arrive)
            if arrive_on > demand.ready_on:
                continue
            next_duration = duration + _route_duration(route)
            if demand.max_duration_hours is not None and next_duration > demand.max_duration_hours:
                continue
            if demand.max_risk is not None and route.risk > demand.max_risk:
                continue
            if not demand.required_capabilities.issubset(route.capabilities):
                continue
            if route_usage.get(route.route_id, route.reserved) >= route.capacity:
                continue
            next_legs = dict(path_legs)
            if route.leg_id not in used_legs and route.leg_id not in next_legs:
                next_legs[route.leg_id] = Leg(route.leg_id, route.origin, route.destination,
                                              route.depart, route.arrive, 1)
            if check_loco and not _loco_ok_with_turn(
                    list(used_legs.values()) + list(next_legs.values()), loco_units, turn_minutes):
                continue
            counter += 1
            heapq.heappush(pq, (
                move_cost + route.unit_move_cost, next_duration, arrive_on, counter,
                route.destination, path + (route,), next_legs,
                visited | {route.destination},
            ))

    return None


def optimize(wagons: Sequence[LeasableWagon], contracts: Sequence[LeaseContract],
             routes: Sequence[RepositionRoute], demands: Sequence[LeaseDemand],
             loco_units: Sequence[Unit] | None = None, turn_minutes: int = 0,
             risk_penalty_rate: float = 1.0) -> LeaseRepositionPlan:
    """Run lease -> maintenance -> location/path -> locomotive checks in priority order."""
    if risk_penalty_rate < 0 or turn_minutes < 0:
        raise ValueError("optimizer parameters")
    for contract in contracts:
        if contract.quantity < 0 or contract.unit_rate < 0:
            raise ValueError("lease contract parameters")
        if contract.reserved > contract.quantity:
            raise ValueError("lease contract over-reserved")
    for route in routes:
        if route.capacity < 0 or route.unit_move_cost < 0 or route.fixed_cost < 0 or route.loco_cost < 0 or route.risk < 0:
            raise ValueError("reposition route parameters")
        if route.reserved > route.capacity:
            raise ValueError("reposition route over-reserved")
    contracts_by_id = {c.contract_id: c for c in contracts}
    wagon_by_id = {w.wagon_id: w for w in wagons}
    contract_ids = {w.contract_id for w in wagons if w.contract_id}
    unknown_contracts = contract_ids.difference(contracts_by_id)
    if unknown_contracts:
        raise ValueError("unknown lease contract: " + ",".join(sorted(unknown_contracts)))
    if any(w.quantity < 0 or w.reserved < 0 or w.reserved > w.quantity for w in wagons):
        raise ValueError("wagon parameters")
    wagon_usage = {w.wagon_id: w.reserved for w in wagons if w.reserved > 0}
    route_usage = {r.route_id: r.reserved for r in routes if r.reserved > 0}
    contract_usage = {c.contract_id: c.reserved for c in contracts
                      if getattr(c, "reserved", 0.0) > 0}
    legs: dict[str, Leg] = {}
    fixed_routes: set[str] = set()
    selections: list[SelectedOption] = []
    traces: list[DemandTrace] = []
    rejected: list[RejectedDemand] = []
    check_loco = loco_units is not None

    for demand in sorted(demands, key=lambda d: (-d.priority, -d.quantity, d.request_id)):
        if demand.quantity <= 0 or demand.lease_days < 0:
            raise ValueError("lease demand")
        service_end = demand.ready_on + timedelta(days=demand.lease_days)
        reasons: dict[str, str] = {}
        stages: list[StageCount] = []

        typed = [w for w in wagons if w.wagon_type == demand.wagon_type]
        free = [w for w in typed if w.quantity - wagon_usage.get(w.wagon_id, 0.0) > 1e-9]
        stages += [StageCount("type", len(typed), len(routes), 0),
                   StageCount("available", len(free), len(routes), 0)]

        eligible = []
        for wagon in free:
            contract = contracts_by_id.get(wagon.contract_id) if wagon.contract_id else None
            ok, reason = _wagon_allowed(wagon, contract, demand, service_end)
            if ok:
                eligible.append(wagon)
            else:
                reasons[wagon.wagon_id] = reason or "ineligible"
        stages += [StageCount("contract", len(eligible), len(routes), 0),
                   StageCount("maintenance", len(eligible), len(routes), 0)]

        candidates: list[SelectedOption] = []
        stage_route_count = 0

        candidate_paths: dict[tuple[str, str | None], tuple[tuple[RepositionRoute, ...], dict[str, Leg]]] = {}
        for wagon in eligible:
            contract = contracts_by_id.get(wagon.contract_id) if wagon.contract_id else None
            paths: list[tuple[tuple[RepositionRoute, ...], dict[str, Leg]]] = []
            if wagon.location == demand.destination:
                paths.append(((), {}))
            built = _build_path(
                wagon.location, demand.destination, routes, demand, loco_units or [],
                turn_minutes, check_loco, route_usage, legs,
                _available_on(wagon, contract, demand.ready_on),
            )
            if built is not None:
                paths.append(built)
                stage_route_count += 1
            for path, path_legs in paths:
                free_wagon = wagon.quantity - wagon_usage.get(wagon.wagon_id, 0.0)
                free_contract = float("inf")
                if contract is not None:
                    free_contract = contract.quantity - contract_usage.get(contract.contract_id, 0.0)
                take = min(demand.quantity, free_wagon, free_contract)
                for route in path:
                    take = min(take, route.capacity - route_usage.get(route.route_id, route.reserved))
                if take <= 1e-9:
                    reasons.setdefault(wagon.wagon_id, "wagon or route capacity exhausted")
                    continue
                fixed_cost = sum(route.fixed_cost for route in path
                                 if route.route_id not in fixed_routes)
                option = _option(demand.request_id, wagon, contract, path, take, demand,
                                 risk_penalty_rate, fixed_cost)
                candidates.append(option)
                candidate_paths[(option.wagon_id, option.route_id)] = (path, path_legs)
        stages.append(StageCount("route", len(eligible), stage_route_count, 0))
        candidates.sort(key=lambda o: (o.marginal_per_wagon, o.duration_hours, o.risk,
                                       o.wagon_id, o.route_id or ""))
        stages.append(StageCount("cost", len(eligible), len(routes), len(candidates)))

        considered = tuple(Alternative(o.wagon_id, o.route_id, True, "available",
                                       o.marginal_per_wagon, o.duration_hours, o.risk)
                           for o in candidates[:5])
        eliminated = tuple(
            Alternative(wagon.wagon_id, None, False, reasons[wagon.wagon_id], 0.0, 0.0, 0.0)
            for wagon in typed[:10] if wagon.wagon_id in reasons
        )[:5]
        remaining = demand.quantity
        for option in candidates:
            if remaining <= 1e-9:
                break
            wagon = wagon_by_id[option.wagon_id]
            contract = contracts_by_id.get(option.contract_id) if option.contract_id else None
            free_wagon = wagon.quantity - wagon_usage.get(option.wagon_id, 0.0)
            free_contract = float("inf") if contract is None else contract.quantity - contract_usage.get(
                contract.contract_id, 0.0)
            path, path_legs = candidate_paths[(option.wagon_id, option.route_id)]
            take = min(remaining, option.quantity, free_wagon, free_contract)
            for route in path:
                take = min(take, route.capacity - route_usage.get(route.route_id, route.reserved))
            if take <= 1e-9:
                continue
            next_route_usage = dict(route_usage)
            next_legs = dict(legs)
            for route in path:
                next_route_usage[route.route_id] = next_route_usage.get(route.route_id, route.reserved) + take
                if route.leg_id not in next_legs:
                    next_legs[route.leg_id] = path_legs[route.leg_id]
            if check_loco and not _loco_ok_with_turn(next_legs.values(), loco_units, turn_minutes):
                reasons.setdefault(option.wagon_id, "locomotive cycle infeasible")
                continue
            fixed_cost = sum(route.fixed_cost for route in path
                             if route.route_id not in fixed_routes)
            selected = _option(demand.request_id, wagon, contract, path, take, demand,
                               risk_penalty_rate, fixed_cost)
            if demand.max_cost_per_wagon is not None:
                existing_qty = sum(item.quantity for item in selections if item.request_id == demand.request_id)
                existing_cost = sum(item.cost.total for item in selections
                                    if item.request_id == demand.request_id)
                if (existing_cost + selected.cost.total) / (existing_qty + take) > demand.max_cost_per_wagon:
                    reasons.setdefault(option.wagon_id, "cost limit")
                    continue
            selections.append(selected)
            wagon_usage[option.wagon_id] = wagon_usage.get(option.wagon_id, 0.0) + take
            if option.contract_id:
                contract_usage[option.contract_id] = contract_usage.get(option.contract_id, 0.0) + take
            route_usage = next_route_usage
            legs = next_legs
            fixed_routes.update(option.route_ids)
            remaining -= take

        if remaining > 1e-9:
            reason = next(iter(reasons.values()), "no feasible lease and reposition option")
            rejected.append(RejectedDemand(demand.request_id, remaining, reason))
            traces.append(DemandTrace(demand.request_id, tuple(stages), considered, eliminated, reason))
        else:
            traces.append(DemandTrace(demand.request_id, tuple(stages), considered, eliminated))

    cycle = None
    if check_loco:
        cycle = circulate(list(legs.values()), list(loco_units), turn_minutes)
    totals = {
        "wagons": sum((s.quantity for s in selections), 0.0),
        "lease_cost": sum((s.cost.lease for s in selections), 0.0),
        "reposition_cost": sum((s.cost.reposition for s in selections), 0.0),
        "locomotive_cost": sum((s.cost.locomotive for s in selections), 0.0),
        "risk_premium": sum((s.cost.risk_premium for s in selections), 0.0),
        "total_cost": sum((s.cost.total for s in selections), 0.0),
    }
    return LeaseRepositionPlan(tuple(selections), tuple(traces), tuple(rejected),
                               dict(route_usage), dict(contract_usage), dict(wagon_usage), cycle, totals)


def _loco_ok_with_turn(legs, loco_units: Sequence[Unit], turn_minutes: int) -> bool:
    try:
        circulate(list(legs), list(loco_units), turn_minutes)
    except ValueError:
        return False
    return True
