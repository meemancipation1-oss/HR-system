"""Session and structured screening regression tests.

Run directly with ``python tests/test_session_store.py`` or via unittest.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.screening_agent import ScreeningOutput
from core.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.sqlite3")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_isolates_and_restores_messages(self) -> None:
        self.store.save_messages("a", [{"role": "user", "content": "candidate A"}])
        self.store.save_messages("b", [{"role": "user", "content": "candidate B"}])

        self.assertEqual(self.store.load_messages("a")[0]["content"], "candidate A")
        self.assertEqual(self.store.load_messages("b")[0]["content"], "candidate B")
        self.assertEqual(self.store.load_messages("missing"), [])

    def test_keeps_recent_history(self) -> None:
        messages = [{"role": "user", "content": str(i)} for i in range(120)]
        self.store.save_messages("a", messages)

        restored = self.store.load_messages("a")
        self.assertEqual(len(restored), 100)
        self.assertEqual(restored[0]["content"], "20")

    def test_persists_and_deletes_task_state(self) -> None:
        state = {
            "current_job": "backend",
            "screenings": {"candidate-1": {"决策": "review"}},
            "selected_candidates": ["candidate-1"],
        }
        self.store.save_task_state("session-a", state)
        self.assertEqual(self.store.load_task_state("session-a"), state)

        self.store.delete("session-a")
        self.assertEqual(self.store.load_task_state("session-a"), {})


class ScreeningOutputTests(unittest.TestCase):
    def test_defaults_to_human_review_without_evidence(self) -> None:
        output = ScreeningOutput(
            评估思路="缺少可审核证据",
            匹配分数=50,
            技能评分=50,
            经验评分=50,
            教育评分=50,
            综合评价="需要复核",
            匹配职位="测试岗位",
            匹配理由="没有足够证据",
        )
        self.assertEqual(output.决策, "review")
        self.assertTrue(output.需要人工复核)
        self.assertEqual(output.证据, [])


if __name__ == "__main__":
    unittest.main()
