"""Regression tests for multi-table candidate retrieval."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import RAGEngine
from core.tools import execute_tool


class MultitableRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        pd.DataFrame(
            [{"人员ID": "1", "姓名": "甲", "学历": "硕士"},
             {"人员ID": "2", "姓名": "乙", "学历": "本科"}]
        ).to_csv(self.root / "SyntheticIndividuals.csv", index=False, encoding="utf-8")
        pd.DataFrame(
            [{"人员ID": "1", "姓名": "甲", "单位": "上海研究所", "职务": "工程师"},
             {"人员ID": "2", "姓名": "乙", "单位": "北京单位", "职务": "助理"}]
        ).to_csv(self.root / "employment.csv", index=False, encoding="utf-8")
        pd.DataFrame(
            [{"人员ID": "1", "姓名": "甲", "语种": "英语", "熟练程度": "精通"},
             {"人员ID": "2", "姓名": "乙", "语种": "法语", "熟练程度": "一般"}]
        ).to_csv(self.root / "LanguageAbility.csv", index=False, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_associated_table_condition_returns_person_id(self) -> None:
        with patch("core.rag.HR_DATA_DIR", self.root):
            engine = object.__new__(RAGEngine)
            employment = engine.search_candidates_in_all_tables("上海研究所", k=10)
            language = engine.search_candidates_in_all_tables("英语 精通", k=10)
        self.assertEqual(employment[0]["人员ID"], "1")
        self.assertIn("employment", employment[0]["matched_tables"])
        self.assertEqual(language[0]["人员ID"], "1")
        self.assertIn("LanguageAbility", language[0]["matched_tables"])

    def test_generic_table_tool_searches_non_person_table(self) -> None:
        with patch("core.rag.HR_DATA_DIR", self.root):
            raw = execute_tool(
                "search_table",
                '{"table_name":"employment","keyword":"上海研究所","k":5}',
            )
        self.assertIn("上海研究所", raw)
        self.assertIn('"人员ID": "1"', raw)


if __name__ == "__main__":
    unittest.main()
