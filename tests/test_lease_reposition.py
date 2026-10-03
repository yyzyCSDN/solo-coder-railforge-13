import unittest
from datetime import date, datetime, timezone

from railforge.api import RailForgeService
from railforge.planning.loco_circulation import Unit
from railforge.storage.ops_store import VersionedStore
from railforge.wagon.lease_batch import BatchConflict, BatchReservation
from railforge.wagon.lease_reposition import (
    LeaseContract,
    LeaseDemand,
    LeasePlanError,
    RepositionRoute,
    contract_key,
    optimize,
    unit_key,
)

U = timezone.utc
ON = date(2026, 2, 1)
DEPART = datetime(2026, 1, 31, 8, tzinfo=U)
NOW = datetime(2026, 1, 30, 12, tzinfo=U)


def contract(cid, location, rate, **kw):
    args = dict(wagon_type="BOX", total_quantity=10, valid_from=date(2026, 1, 1),
                valid_to=date(2026, 12, 31), maintenance_due=date(2026, 6, 1))
    args.update(kw)
    return LeaseContract(cid, args["wagon_type"], location, args["total_quantity"],
                         args["valid_from"], args["valid_to"], rate,
                         args["maintenance_due"], args.get("leased_quantity", 0),
                         args.get("allowed_destinations", frozenset()))


def pool():
    return [
        contract("C1", "YARD-B", 30.0),                                    # direct supply, pricier
        contract("C2", "YARD-A", 20.0),                                    # needs empty move, cheaper
        contract("C3", "YARD-B", 10.0, maintenance_due=date(2026, 2, 5)),  # maintenance due
        contract("C4", "YARD-B", 5.0, valid_to=date(2026, 2, 5)),          # window too short
        contract("C6", "YARD-B", 1.0, allowed_destinations=frozenset({"YARD-C"})),
        contract("C7", "YARD-D", 5.0),                                     # no path to YARD-B
    ]


def routes():
    return [
        RepositionRoute("R1", "YARD-A", "YARD-B", 10, 40.0, 1.0, 0.1),
        RepositionRoute("R2", "YARD-A", "YARD-B", 10, 1.0, 5.0, 0.0),  # arrives too late
    ]


def demand():
    return LeaseDemand("D1", "BOX", "YARD-B", ON, 5, 10, priority=1)


class OptimizerTests(unittest.TestCase):
    def test_multi_stage_selection_and_traceable_tradeoffs(self):
        units = [Unit("U1", "YARD-A", datetime(2026, 1, 30, tzinfo=U), 100.0)]
        plan = optimize(pool(), [demand()], routes(), units, DEPART,
                        turn_minutes=60, risk_weight=1.0, loco_hourly_cost=10.0)
        self.assertEqual(len(plan.assignments), 1)
        chosen = plan.assignments[0]
        self.assertEqual(chosen.contract_id, "C2")
        self.assertEqual(chosen.route_id, "R1")
        self.assertEqual(chosen.legs, ("D1:R1",))
        self.assertEqual(chosen.units, ("U1",))
        self.assertAlmostEqual(chosen.cost.lease, 1000.0)
        self.assertAlmostEqual(chosen.cost.reposition, 200.0)
        self.assertAlmostEqual(chosen.cost.loco, 240.0)
        self.assertAlmostEqual(plan.total_cost, 1440.1)

        stages = {(t.contract_id, t.stage, t.outcome) for t in plan.trace}
        self.assertIn(("C3", "maintenance", "rejected"), stages)
        self.assertIn(("C4", "contract", "rejected"), stages)
        self.assertIn(("C6", "contract", "rejected"), stages)
        self.assertIn(("C7", "location", "rejected"), stages)
        self.assertIn(("C2", "cost", "chosen"), stages)
        costlier = next(t for t in plan.trace if t.contract_id == "C1" and t.stage == "cost")
        self.assertEqual(costlier.outcome, "rejected")
        self.assertAlmostEqual(costlier.cost, 1500.0)
        late = next(t for t in plan.trace if t.route_id == "R2")
        self.assertEqual(late.stage, "route")
        self.assertEqual(late.reason, "no time to reposition before need date")

        self.assertEqual(plan.contract_updates["C2"].leased_quantity, 5)
        self.assertNotIn("C1", plan.contract_updates)
        moved = plan.unit_updates["U1"]
        self.assertEqual(moved.location, "YARD-B")
        self.assertAlmostEqual(moved.due_hours, 76.0)
        self.assertEqual(moved.available_at, datetime(2026, 2, 2, 1, tzinfo=U))

    def test_loco_cycle_infeasible_falls_back_to_direct_supply(self):
        units = [Unit("U1", "YARD-A", datetime(2026, 1, 30, tzinfo=U), 10.0)]  # due before leg ends
        plan = optimize(pool(), [demand()], routes(), units, DEPART,
                        turn_minutes=60, loco_hourly_cost=10.0)
        chosen = plan.assignments[0]
        self.assertEqual(chosen.contract_id, "C1")
        self.assertIsNone(chosen.route_id)
        self.assertEqual(chosen.legs, ())
        self.assertEqual(plan.unit_updates, {})
        self.assertIn(("C2", "loco", "rejected"),
                      {(t.contract_id, t.stage, t.outcome) for t in plan.trace})

    def test_infeasible_demand_raises_with_trace(self):
        demands = [LeaseDemand("D9", "TANK", "YARD-B", ON, 2, 5)]
        with self.assertRaises(LeasePlanError) as ctx:
            optimize(pool(), demands, routes(), [], DEPART)
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertEqual(ctx.exception.demand_id, "D9")
        self.assertTrue(all(t.stage == "contract" for t in ctx.exception.trace))

    def test_one_locomotive_cycles_through_consecutive_demands(self):
        contracts = [
            contract("CA", "A", 20.0),                          # BOX supply at A
            contract("CB", "B", 20.0, wagon_type="FLAT"),       # FLAT supply at B
        ]
        demands = [
            LeaseDemand("D1", "BOX", "B", date(2026, 2, 1), 3, 10, priority=2),
            LeaseDemand("D2", "FLAT", "C", date(2026, 2, 3), 3, 10, priority=1),
        ]
        legs = [
            RepositionRoute("RAB", "A", "B", 10, 40.0, 1.0, 0.1),
            RepositionRoute("RBC", "B", "C", 10, 40.0, 1.0, 0.1),
        ]
        units = [Unit("U1", "A", datetime(2026, 1, 30, tzinfo=U), 200.0)]
        plan = optimize(contracts, demands, legs, units, DEPART,
                        turn_minutes=60, loco_hourly_cost=10.0)
        self.assertEqual([a.route_id for a in plan.assignments], ["RAB", "RBC"])
        self.assertTrue(all(a.units == ("U1",) for a in plan.assignments))
        cycled = plan.unit_updates["U1"]
        self.assertEqual(cycled.location, "C")
        self.assertAlmostEqual(cycled.due_hours, 152.0)  # 200 - 24 - 24


class BatchReservationTests(unittest.TestCase):
    def test_conflict_rolls_back_and_retry_with_fresh_versions_commits(self):
        store = VersionedStore()
        store.write("k1", "v1")
        store.write("k2", "v2")
        batch = BatchReservation(store)
        batch.stage("k1", "new1", 1).stage("k2", "new2", 1)
        store.write("k2", "other", 1)  # concurrent writer wins k2 first
        with self.assertRaises(BatchConflict) as ctx:
            batch.commit("op-x")
        self.assertEqual(ctx.exception.key, "k2")
        self.assertEqual(ctx.exception.rolled_back, ("k1",))
        self.assertEqual(store.read("k1").value, "v1")      # rolled back
        self.assertEqual(store.read("k2").value, "other")   # untouched

        batch.stage("k1", "new1", store.read("k1").version)
        batch.stage("k2", "new2", store.read("k2").version)
        receipt = batch.commit("op-x")
        self.assertEqual(receipt.versions, {"k1": 4, "k2": 3})
        self.assertEqual(store.read("k1").value, "new1")
        self.assertEqual(store.read("k2").value, "new2")

    def test_commit_is_idempotent_per_operation_id(self):
        store = VersionedStore()
        batch = BatchReservation(store)
        batch.stage("a", 1)
        first = batch.commit("op-1")
        second = batch.commit("op-1")
        self.assertIs(first, second)
        self.assertEqual(store.read("a").version, 1)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.service = RailForgeService()
        self.store = self.service.workflow().store
        self.c2 = contract("C2", "YARD-A", 20.0)
        self.u1 = Unit("U1", "YARD-A", datetime(2026, 1, 30, tzinfo=U), 100.0)
        self.store.write(contract_key("C2"), self.c2)
        self.store.write(unit_key("U1"), self.u1)
        self.expected = {contract_key("C2"): 1, unit_key("U1"): 1}

    def call_feature(self, operation_id, expected):
        return self.service.features.optimize_lease_reposition(
            operation_id, [self.c2], [demand()], routes()[:1], [self.u1],
            DEPART, 60, expected, NOW, loco_hourly_cost=10.0)

    def test_atomic_commit_and_idempotent_retry(self):
        receipt = self.call_feature("op-lease", self.expected)
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        plan = receipt.result["plan"]
        self.assertEqual(plan.assignments[0].contract_id, "C2")
        self.assertEqual(receipt.result["versions"],
                         {contract_key("C2"): 2, unit_key("U1"): 2})
        leased = self.store.read(contract_key("C2"))
        self.assertEqual(leased.value.leased_quantity, 5)
        self.assertEqual(leased.version, 2)
        moved = self.store.read(unit_key("U1"))
        self.assertEqual(moved.value.location, "YARD-B")
        self.assertEqual(moved.version, 2)
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())

        again = self.call_feature("op-lease", self.expected)
        self.assertIs(again, receipt)
        self.assertEqual(self.store.read(contract_key("C2")).version, 2)

    def test_stale_versions_conflict_roll_back_then_retry(self):
        stale = {contract_key("C2"): 1, unit_key("U1"): 99}
        with self.assertRaises(BatchConflict):
            self.call_feature("op-r", stale)
        rolled = self.store.read(contract_key("C2"))
        self.assertEqual(rolled.value, self.c2)  # staged write rolled back
        self.assertEqual(rolled.version, 3)
        self.assertEqual(self.store.read(unit_key("U1")).version, 1)
        self.assertNotIn("op-r", self.service.workflow().receipts)

        fresh = {contract_key("C2"): 3, unit_key("U1"): 1}
        receipt = self.call_feature("op-r", fresh)
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(self.store.read(contract_key("C2")).value.leased_quantity, 5)
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())


if __name__ == "__main__":
    unittest.main()
