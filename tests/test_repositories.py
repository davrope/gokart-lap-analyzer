from __future__ import annotations

import unittest

from services.repositories import AttemptRepository, TrackRepository


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, store: dict[str, list[dict]]):
        self.table_name = table_name
        self.store = store
        self.filters = []
        self.payload = None
        self.action = "select"

    def select(self, *_args, **_kwargs):
        self.action = "select"
        return self

    def insert(self, payload):
        self.action = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.action = "update"
        self.payload = payload
        return self

    def delete(self):
        self.action = "delete"
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def in_(self, key, values):
        self.filters.append((key, set(values)))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = list(self.store.get(self.table_name, []))

        def _match(row):
            for k, v in self.filters:
                if isinstance(v, set):
                    if row.get(k) not in v:
                        return False
                else:
                    if row.get(k) != v:
                        return False
            return True

        if self.action == "select":
            return _FakeResponse([r for r in rows if _match(r)])

        if self.action == "insert":
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            self.store.setdefault(self.table_name, []).extend(payloads)
            return _FakeResponse(payloads)

        if self.action == "update":
            out = []
            for r in self.store.get(self.table_name, []):
                if _match(r):
                    r.update(self.payload)
                    out.append(dict(r))
            return _FakeResponse(out)

        if self.action == "delete":
            kept = []
            deleted = []
            for r in self.store.get(self.table_name, []):
                if _match(r):
                    deleted.append(r)
                else:
                    kept.append(r)
            self.store[self.table_name] = kept
            return _FakeResponse(deleted)

        return _FakeResponse([])


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        return _FakeQuery(name, self.store)


class RepositoriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = {
            "tracks": [],
            "attempts": [],
            "attempt_laps": [],
            "attempt_curves": [],
        }
        self.client = _FakeClient(self.store)
        self.track_repo = TrackRepository(self.client)
        self.attempt_repo = AttemptRepository(self.client)

    def test_create_and_list_track(self) -> None:
        t = self.track_repo.create_track("u1", "Kartodromo", "cw", None, None)
        self.assertEqual(t["name"], "Kartodromo")
        tracks = self.track_repo.list_tracks("u1")
        self.assertEqual(len(tracks), 1)

    def test_create_update_and_get_attempt(self) -> None:
        created = self.attempt_repo.create_attempt(
            {
                "id": "a1",
                "user_id": "u1",
                "track_id": "t1",
                "source_filename": "s.fit",
                "storage_bucket": "fit-files",
                "storage_path": "u1/a1/s.fit",
                "status": "uploaded",
                "method_name": "GPS Gate (fast-points + distance minima)",
                "params_json": {},
            }
        )
        self.assertEqual(created["id"], "a1")

        updated = self.attempt_repo.update_attempt_processed("a1", {"status": "processed", "laps_detected": 12})
        self.assertEqual(updated["status"], "processed")

        got = self.attempt_repo.get_attempt("a1")
        self.assertIsNotNone(got)
        self.assertEqual(got["laps_detected"], 12)

    def test_replace_laps_curves(self) -> None:
        self.attempt_repo.create_attempt(
            {
                "id": "a2",
                "user_id": "u1",
                "track_id": "t1",
                "source_filename": "s.fit",
                "storage_bucket": "fit-files",
                "storage_path": "u1/a2/s.fit",
                "status": "uploaded",
                "method_name": "GPS Gate (fast-points + distance minima)",
                "params_json": {},
            }
        )
        self.attempt_repo.replace_attempt_laps("a2", [{"attempt_id": "a2", "lap": 1}, {"attempt_id": "a2", "lap": 2}])
        self.attempt_repo.replace_attempt_curves("a2", [{"attempt_id": "a2", "lap": 1, "curve_id": 1}])

        self.assertEqual(len(self.store["attempt_laps"]), 2)
        self.assertEqual(len(self.store["attempt_curves"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
