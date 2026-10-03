import unittest
from datetime import date, datetime, timedelta, timezone

from railforge.core.versioning import VersionConflict
from railforge.planning.loco_circulation import Unit
from railforge.api import RailForgeService
from railforge.wagon.lease_reposition import (
    LeaseContract,
    LeaseDemand,
    LeasableWagon,
    RepositionRoute,
    optimize,
)
from railforge.wagon.lease_reservation import LeaseReservationStore, commit_batch

U = timezone.utc


class LeaseRepositionTests(unittest.TestCase):
    def setUp(self):
        self.ready = date(2026, 2, 10)
        self.contracts = [
            LeaseContract("C-CHEAP", "north", "A", "BOX", 2, date(2026, 2, 1),
                          expires_on=date(2026, 3, 1), unit_rate=10),
            LeaseContract("C-RESTRICTED", "east", "B", "BOX", 1, date(2026, 2, 1),
                          unit_rate=8, restrictions=frozenset({"hazard"})),
        ]
        self.wagons = [
            LeasableWagon("W-LOCAL", None, "BOX", "D", date(2026, 3, 1), quantity=1),
            LeasableWagon("W-A", "C-CHEAP", "BOX", "A", date(2026, 3, 1), quantity=2),
            LeasableWagon("W-B", "C-RESTRICTED", "BOX", "B", date(2026, 2, 12), quantity=1),
        ]
        self.routes = [
            RepositionRoute(
                "R-A-D", "A", "D",
                datetime(2026, 2, 8, 8, tzinfo=U),
                datetime(2026, 2, 9, 8, tzinfo=U),
                capacity=2, leg_id="L-A-D", unit_move_cost=5, fixed_cost=20, loco_cost=2, risk=0.1,
            )
        ]
        self.loco_units = [
            Unit("LOCO-1", "A", datetime(2026, 2, 8, tzinfo=U), due_hours=100),
        ]

    def test_multi_stage_plan_chooses_cheapest_feasible_wagon_and_route(self):
        demand = LeaseDemand("D1", "BOX", "D", 2, self.ready, lease_days=5)
        plan = optimize(self.wagons, self.contracts, self.routes, [demand], self.loco_units, 30, 1)
        self.assertEqual(plan.rejected, ())
        self.assertEqual([s.wagon_id for s in plan.selections], ["W-LOCAL", "W-A"])
        moved = next(s for s in plan.selections if s.wagon_id == "W-A")
        self.assertEqual(moved.route_id, "R-A-D")
        self.assertEqual(moved.route_ids, ("R-A-D",))
        self.assertGreater(moved.cost.total, 0)
        self.assertEqual(plan.route_usage["R-A-D"], 1)
        self.assertIn("L-A-D", plan.cycle[0])
        self.assertGreater(plan.totals["total_cost"], 0)
        stages = {s.stage for s in plan.traces[0].stages}
        self.assertEqual(stages, {"type", "available", "contract", "maintenance", "route", "cost"})
        self.assertTrue(plan.traces[0].considered)

    def test_multi_leg_empty_return_path_is_reserved_and_circulated(self):
        routes = [
            RepositionRoute("R-A-X", "A", "X",
                            datetime(2026, 2, 8, 8, tzinfo=U),
                            datetime(2026, 2, 8, 18, tzinfo=U),
                            capacity=1, leg_id="L-A-X", unit_move_cost=1, fixed_cost=2),
            RepositionRoute("R-X-D", "X", "D",
                            datetime(2026, 2, 9, 8, tzinfo=U),
                            datetime(2026, 2, 9, 18, tzinfo=U),
                            capacity=1, leg_id="L-X-D", unit_move_cost=1, fixed_cost=2),
        ]
        demand = LeaseDemand("ML", "BOX", "D", 1, self.ready, lease_days=5)
        plan = optimize([self.wagons[1]], [self.contracts[0]], routes, [demand],
                        self.loco_units, 600, 1)
        self.assertEqual(plan.rejected, ())
        selected = plan.selections[0]
        self.assertEqual(selected.route_ids, ("R-A-X", "R-X-D"))
        self.assertEqual(selected.route_id, "R-X-D")
        self.assertEqual(plan.route_usage["R-A-X"], 1)
        self.assertEqual(plan.route_usage["R-X-D"], 1)
        self.assertEqual(set(plan.cycle[0]), {"L-A-X", "L-X-D"})

    def test_trace_records_maintenance_and_restriction_eliminations(self):
        demand = LeaseDemand("TRACE", "BOX", "D", 3, self.ready, lease_days=20,
                             forbidden_restrictions=frozenset({"hazard"}))
        plan = optimize(self.wagons, self.contracts, self.routes, [demand], self.loco_units)
        reasons = {a.wagon_id: a.reason for a in plan.traces[0].eliminated}
        self.assertIn("W-B", reasons)
        self.assertEqual(reasons["W-B"], "forbidden contract restriction")
        self.assertTrue(any("maintenance" in reason for reason in reasons.values()))

    def test_maintenance_contract_restriction_and_loco_cycle_are_enforced(self):
        demands = [
            LeaseDemand("HAZ", "BOX", "D", 1, self.ready,
                        forbidden_restrictions=frozenset({"hazard"})),
        ]
        plan = optimize(self.wagons, self.contracts, self.routes, demands, self.loco_units)
        self.assertEqual([s.wagon_id for s in plan.selections], ["W-LOCAL"])

        no_loco = [Unit("LOCO-X", "X", datetime(2026, 2, 8, tzinfo=U), due_hours=100)]
        demands = [LeaseDemand("MOVE", "BOX", "D", 1, self.ready)]
        plan = optimize(self.wagons, self.contracts, self.routes, demands, no_loco)
        self.assertEqual([s.wagon_id for s in plan.selections], ["W-LOCAL"])

    def test_batch_reservation_cas_rollback_and_idempotent_retry(self):
        demand = LeaseDemand("B1", "BOX", "D", 2, self.ready, lease_days=1)
        plan = optimize(self.wagons, self.contracts, self.routes, [demand], self.loco_units)
        store = LeaseReservationStore(self.wagons, self.routes, self.contracts)
        committed = store.compare_and_swap("OP-1", plan, 0)
        self.assertEqual(committed.version, 1)
        self.assertEqual(store.snapshot().version, 1)
        self.assertEqual(store.get("OP-1"), committed)

        with self.assertRaises(VersionConflict):
            store.compare_and_swap("OP-CONFLICT", plan, 0)
        self.assertEqual(store.snapshot().version, 1)
        self.assertEqual(store.get("OP-CONFLICT"), None)

        retry = store.reserve("OP-1", plan, 0)
        self.assertIs(retry, committed)
        self.assertEqual(store.snapshot().version, 1)

        bad_plan = optimize(
            self.wagons, self.contracts, self.routes,
            [LeaseDemand("B2", "BOX", "D", 10, self.ready)], None,
        )
        self.assertTrue(bad_plan.rejected)
        with self.assertRaises(ValueError):
            store.reserve("OP-FAIL", bad_plan, 1)
        self.assertEqual(store.snapshot().version, 1)
        self.assertIsNone(store.get("OP-FAIL"))

    def test_commit_batch_reuses_catalog_state_on_idempotent_retry(self):
        store = LeaseReservationStore()
        demand = LeaseDemand("C1", "BOX", "D", 2, self.ready, lease_days=1)
        first = commit_batch(store, "C-OP", self.wagons, self.contracts, self.routes,
                             [demand], self.loco_units, 30, 1)
        self.assertEqual(first.version, 1)
        second = commit_batch(store, "C-OP", self.wagons, self.contracts, self.routes,
                              [demand], self.loco_units, 30, 1, expected_version=1)
        self.assertIs(second, first)
        self.assertEqual(store.snapshot().version, 1)
        self.assertEqual(store.snapshot().value.wagon_reserved["W-A"], 1)
        self.assertEqual(store.overloaded(), ())

    def test_domain_commitment_rollback_allows_retry(self):
        demand = LeaseDemand("R1", "BOX", "D", 1, self.ready, lease_days=1)
        plan = optimize(self.wagons, self.contracts, self.routes, [demand], self.loco_units)
        store = LeaseReservationStore(self.wagons, self.routes, self.contracts)
        committed = store.reserve("R-OP", plan)
        self.assertEqual(store.snapshot().version, 1)
        rolled_back = store.rollback_commitment("R-OP")
        self.assertTrue(rolled_back.rolled_back)
        self.assertEqual(store.snapshot().version, 0)
        self.assertIsNone(store.get("R-OP"))
        retried = store.reserve("R-OP", plan)
        self.assertEqual(retried.version, 1)

    def test_feature_records_cost_trace_and_is_idempotent(self):
        service = RailForgeService()
        at = datetime(2026, 2, 1, tzinfo=U)
        demand = LeaseDemand("F1", "BOX", "D", 2, self.ready, lease_days=2)
        receipt = service.features.lease_and_reposition_wagons(
            "F-OP", self.wagons, self.contracts, self.routes, [demand], self.loco_units, 30, 1, at
        )
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.result["version"], 1)
        self.assertIn("totals", receipt.result["cost_trace"])
        self.assertEqual(receipt.result["cost_trace"]["requests"]["F1"]["rejected_reason"], None)
        retry = service.features.lease_and_reposition_wagons(
            "F-OP", self.wagons, self.contracts, self.routes, [demand], self.loco_units, 30, 1, at
        )
        self.assertIs(retry, receipt)
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_feature_stale_version_does_not_partially_commit(self):
        service = RailForgeService()
        at = datetime(2026, 2, 1, tzinfo=U)
        demand = LeaseDemand("V1", "BOX", "D", 1, self.ready)
        with self.assertRaises(VersionConflict):
            service.features.lease_and_reposition_wagons(
                "STALE", self.wagons, self.contracts, self.routes, [demand],
                self.loco_units, 30, 1, at, expected_version=99
            )
        self.assertIsNone(service.workflow().lease_reservations.get("STALE"))
        self.assertEqual(service.workflow().lease_reservations.snapshot().version, 0)
        self.assertEqual(service.workflow().stream.head, 0)


if __name__ == "__main__":
    unittest.main()
