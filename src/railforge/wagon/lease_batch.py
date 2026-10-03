from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from railforge.storage.ops_store import VersionedStore


class BatchConflict(RuntimeError):
    """A staged compare-and-swap failed; every write from the attempt was rolled back."""

    def __init__(self, key: str, rolled_back):
        self.key = key
        self.rolled_back = tuple(rolled_back)
        super().__init__(f"version conflict at {key}; rolled back {len(self.rolled_back)} staged writes")


@dataclass(frozen=True)
class BatchReceipt:
    operation_id: str
    versions: Mapping[str, int]


class BatchReservation:
    """Compare-and-swap batch writer with rollback and idempotent retry.

    stage() collects writes; commit() applies them against the store in one
    pass. A version conflict on any key rolls back the keys already written
    by the attempt (a rolled-back key that was absent before the batch reads
    back as None), and commit() may be retried once expected versions are
    refreshed. Committing again with the same operation_id returns the
    original receipt, so callers can retry safely after a committed batch.
    """

    def __init__(self, store: VersionedStore):
        self._store = store
        self._staged: dict[str, tuple[Any, int | None]] = {}
        self._committed: dict[str, BatchReceipt] = {}

    @property
    def staged(self) -> Mapping[str, Any]:
        return {key: value for key, (value, _) in self._staged.items()}

    @property
    def committed(self) -> Mapping[str, BatchReceipt]:
        return dict(self._committed)

    def stage(self, key: str, value: Any, expected_version: int | None = None):
        if expected_version is not None and expected_version < 0:
            raise ValueError("expected_version")
        self._staged[key] = (value, expected_version)
        return self

    def commit(self, operation_id: str) -> BatchReceipt:
        previous = self._committed.get(operation_id)
        if previous is not None:
            return previous
        prior = {key: self._store.read(key) for key in self._staged}
        written = []
        try:
            for key, (value, expected) in self._staged.items():
                version = self._store.write(key, value, expected)
                written.append((key, version))
        except RuntimeError:
            self._rollback(prior, written)
            raise BatchConflict(key, [k for k, _ in written]) from None
        receipt = BatchReceipt(operation_id, {key: version for key, version in written})
        self._committed[operation_id] = receipt
        self._staged.clear()
        return receipt

    def _rollback(self, prior, written):
        for key, version in reversed(written):
            before = prior[key]
            self._store.write(key, before.value if before is not None else None,
                              expected_version=version)
