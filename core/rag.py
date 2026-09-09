"""RAG and structured retrieval over the HR CSV dataset.

The candidate vector document is assembled from every CSV that has a
``人员ID`` column. This keeps one retrievable document per candidate while
retaining source table labels for evidence and debugging. Tables without
``人员ID`` remain available through the structured table-search API.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from chromadb.api.types import EmbeddingFunction

from config.settings import HR_DATA_DIR, VECTOR_DB_DIR
from core.llm import embed_texts


class SentenceTransformerEmbeddingFn(EmbeddingFunction):
    """Wrap the local sentence-transformers model for ChromaDB."""

    def __init__(self) -> None:
        pass

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)


class RAGEngine:
    """Candidate-level vector search plus exact retrieval over every CSV."""

    MAX_BATCH_SIZE = 5000
    INDEX_VERSION = "2-multitable-person"
    _TABLE_CACHE: dict[Path, tuple[int, int, pd.DataFrame]] = {}
    _QUERY_STOPWORDS = {
        "帮我", "请", "寻找", "找出", "查询", "筛选", "搜索", "候选人", "人员",
        "符合", "适合", "需要", "要求", "具有", "拥有", "会", "能", "的", "在",
        "过", "吗", "哪些", "所有", "一些", "任职", "工作", "经历", "请问",
        "想要", "一个", "几个", "有", "且", "并且", "以及", "同时", "或者", "和",
    }
    _STRUCTURED_HINTS = (
        "语言", "语种", "英语", "日语", "韩语", "法语", "德语", "俄语", "西班牙语",
        "熟练", "认证", "任职", "工作经历", "单位", "职位", "职务", "专利", "论文",
        "著作", "软著", "软件著作", "项目", "奖励", "获奖", "表彰", "演习", "作战",
        "军事", "飞行", "舰艇", "武器", "体检", "健康", "涉密", "奖惩", "团体",
        "兼职", "待遇", "标准规范", "人才认定",
    )
    _TABLE_HINTS = {
        "LanguageAbility": ("语言", "语种", "英语", "日语", "韩语", "法语", "德语", "俄语", "熟练", "认证"),
        "employment": ("任职", "工作经历", "任职单位", "单位", "职位", "职务"),
        "PatentInvention": ("专利", "发明"),
        "Writing": ("论文", "著作", "发表"),
        "SoftwareCopyright": ("软著", "软件著作"),
        "MajorSciProject": ("项目", "科研"),
        "TechnicalAward": ("奖励", "获奖", "科技奖"),
        "TechnicalTalent": ("人才认定", "人才类别"),
        "TechnicalTalentAward": ("表彰", "人才表彰"),
        "MilitaryExercise": ("演习", "演训"),
        "MilitaryCompetitionExperience": ("比武", "竞赛"),
        "MilitaryExperience": ("军事经历", "军衔", "职务层级"),
        "War": ("作战", "参战"),
        "NonWar": ("非战争", "行动"),
        "FlightExperience": ("飞行", "飞机", "飞行时间"),
        "CaptainExperience": ("舰艇", "舰长"),
        "RocketForceExperience": ("火箭军", "导弹"),
        "WeaponEquipment": ("武器", "装备", "操作水平"),
        "EvaluationInfo": ("考核", "评价"),
        "AwardPunishment": ("奖惩", "惩处"),
        "PhysicalExamInfo": ("体检", "健康", "病史"),
        "ConfidentialStatus": ("涉密", "政审"),
        "GroupParticipation": ("团体", "组织参与"),
        "SecondaryAppointment": ("兼职", "地方兼职"),
        "TreatmentLv": ("待遇", "工资", "住房待遇"),
        "StandardSpecification": ("标准规范", "标准"),
        "MedalInformation": ("勋章", "奖章"),
        "PersonalDynamicStatus": ("在位", "动态", "出发地点", "目标地点"),
        "MoralityIntegrityPerformance": ("德才", "品德", "鉴定"),
    }

    def __init__(self, collection_name: str = "hr_candidates") -> None:
        self._collection_name = collection_name
        self._embed_fn = SentenceTransformerEmbeddingFn()
        self._client = chromadb.PersistentClient(str(VECTOR_DB_DIR))
        self._collection = None
        self._initialized = False
        self._person_documents: dict[str, str] = {}
        self._person_names: dict[str, str] = {}

    def build_index(self, force: bool = False) -> None:
        """Build one candidate document from every CSV containing ``人员ID``."""
        if self._initialized and not force:
            return

        if force:
            self._delete_collection()

        if self._collection is None:
            try:
                existing = self._client.get_collection(
                    self._collection_name,
                    embedding_function=self._embed_fn,
                )
                if (existing.metadata or {}).get("schema_version") != self.INDEX_VERSION:
                    self._delete_collection()
                else:
                    self._collection = existing
                    self._initialized = True
                    return
            except Exception:
                self._collection = None

        self._collection = self._client.create_collection(
            self._collection_name,
            metadata={
                "schema_version": self.INDEX_VERSION,
                "source": "all_csv_tables_with_person_id",
            },
            embedding_function=self._embed_fn,
        )

        person_parts: dict[str, list[str]] = defaultdict(list)
        person_names: dict[str, str] = {}
        person_tables: dict[str, set[str]] = defaultdict(set)
        for path in sorted(HR_DATA_DIR.glob("*.csv")):
            df = self._read_csv(path, use_cache=False)
            if df is None or "人员ID" not in df.columns:
                continue
            table_name = path.stem
            for row_number, (_, row) in enumerate(df.iterrows()):
                person_id = str(row.get("人员ID", "")).strip()
                if not person_id or person_id.lower() == "nan":
                    continue
                if not person_names.get(person_id):
                    name = str(row.get("姓名", "")).strip()
                    if name and name.lower() != "nan":
                        person_names[person_id] = name
                row_text = self._row_to_document(row)
                if row_text:
                    person_tables[person_id].add(table_name)
                    person_parts[person_id].append(
                        f"[来源表: {table_name}; 行号: {row_number + 2}]\n{row_text}"
                    )

        all_ids = sorted(person_parts)
        all_documents = [
            f"人员ID: {person_id}\n姓名: {person_names.get(person_id, '')}\n"
            + "\n\n".join(person_parts[person_id])
            for person_id in all_ids
        ]
        all_metadatas = [
            {
                "entity_type": "person",
                "人员ID": person_id,
                "姓名": person_names.get(person_id, ""),
                "source_tables": ",".join(sorted(person_tables[person_id])),
            }
            for person_id in all_ids
        ]

        total = len(all_ids)
        print(
            f"  共 {total} 名候选人，聚合 "
            f"{len({table for tables in person_tables.values() for table in tables})} 张人员表，"
            "分批写入..."
        )
        for start in range(0, total, self.MAX_BATCH_SIZE):
            end = min(start + self.MAX_BATCH_SIZE, total)
            self._collection.add(
                ids=[f"person:{person_id}" for person_id in all_ids[start:end]],
                documents=all_documents[start:end],
                metadatas=all_metadatas[start:end],
            )
            print(f"  已写入 {end}/{total} 条", end="\r")
        print(f"\n  写入完成！共 {total} 名候选人。")
        self._person_documents = dict(zip(all_ids, all_documents))
        self._person_names = person_names
        self._initialized = True

    def _delete_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        self._person_documents.clear()
        self._person_names.clear()

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return candidates from the multi-table candidate vector index."""
        if not self._initialized:
            self.build_index()

        limit = max(k, min(k * 3, 50))
        results = self._collection.query(
            query_texts=[query], n_results=limit, where={"entity_type": "person"}
        )
        vector_ids = results.get("ids", [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        output = []
        for index in range(min(k, len(vector_ids))):
            metadata = metadatas[index] or {}
            output.append({
                "人员ID": self._metadata_person_id(results, index),
                "姓名": str(metadata.get("姓名", "")),
                "摘要": (documents[index] or "")[:400] + "...",
                "distance": distances[index] if index < len(distances) else 0,
                "source": "multitable_rag",
            })
        return output

    def search_table(self, table_name: str, keyword: str, k: int = 20) -> list[dict[str, Any]]:
        """Search any CSV table by substring and return raw matching rows."""
        allowed = {path.stem: path for path in HR_DATA_DIR.glob("*.csv")}
        path = allowed.get(table_name)
        if path is None:
            raise ValueError(f"未知数据表: {table_name}")
        df = self._read_csv(path, use_cache=True)
        if df is None or df.empty:
            return []
        needle = str(keyword).strip().lower()
        if not needle:
            return []
        mask = df.astype(str).apply(
            lambda column: column.str.lower().str.contains(needle, regex=False, na=False)
        ).any(axis=1)
        return [
            {str(key): ("" if pd.isna(value) else str(value)) for key, value in row.items()}
            for _, row in df[mask].head(k).iterrows()
        ]

    def search_candidates_in_all_tables(self, keyword: str, k: int = 20) -> list[dict]:
        """Recall candidate IDs by scanning every person-related CSV."""
        terms = self._extract_query_terms(keyword)
        if not terms:
            return []
        hits: dict[str, dict[str, Any]] = {}
        for path in self._candidate_search_paths(keyword):
            df = self._read_csv(path, use_cache=True)
            if df is None or "人员ID" not in df.columns:
                continue
            searchable = df.fillna("").astype(str)
            scores = pd.Series(0, index=df.index, dtype="int64")
            for term in terms:
                term_mask = searchable.apply(
                    lambda column: column.str.contains(term, case=False, regex=False, na=False)
                ).any(axis=1)
                scores = scores.add(term_mask.astype("int64"), fill_value=0)
            for row_index in scores[scores > 0].index:
                row = df.loc[row_index]
                person_id = str(row.get("人员ID", "")).strip()
                if not person_id or person_id.lower() == "nan":
                    continue
                item = hits.setdefault(
                    person_id,
                    {
                        "人员ID": person_id,
                        "姓名": str(row.get("姓名", "")),
                        "摘要": "",
                        "matched_tables": [],
                        "match_score": 0,
                    },
                )
                item["match_score"] += int(scores.loc[row_index])
                if path.stem not in item["matched_tables"]:
                    item["matched_tables"].append(path.stem)
                if not item["摘要"]:
                    item["摘要"] = self._row_to_document(row)[:400]
        return sorted(hits.values(), key=lambda item: (-item["match_score"], item["人员ID"]))[:k]

    @classmethod
    def has_structured_hint(cls, query: str) -> bool:
        """Whether a query likely contains an exact/associated-table condition."""
        text = str(query)
        return any(hint in text for hint in cls._STRUCTURED_HINTS)

    @classmethod
    def _candidate_search_paths(cls, query: str) -> list[Path]:
        """Select relevant person tables; fall back to all person tables."""
        text = str(query)
        names = {
            table_name
            for table_name, hints in cls._TABLE_HINTS.items()
            if any(hint in text for hint in hints)
        }
        available = {path.stem: path for path in HR_DATA_DIR.glob("*.csv")}
        if not names:
            names = set()
            for name, path in available.items():
                try:
                    if "人员ID" in pd.read_csv(path, encoding="utf-8", nrows=0).columns:
                        names.add(name)
                except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError):
                    continue
        return [available[name] for name in sorted(names) if name in available]

    def _load_person_cache(self) -> None:
        if self._person_documents or not self._collection:
            return
        data = self._collection.get(where={"entity_type": "person"}, include=["documents", "metadatas"])
        for document, metadata in zip(data.get("documents", []), data.get("metadatas", [])):
            metadata = metadata or {}
            person_id = str(metadata.get("人员ID", "")).strip()
            if person_id:
                self._person_documents[person_id] = document or ""
                self._person_names[person_id] = str(metadata.get("姓名", ""))

    @staticmethod
    def _metadata_person_id(results: dict, index: int) -> str:
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        if index < len(metadatas) and metadatas[index]:
            value = metadatas[index].get("人员ID", "")
            if value:
                return str(value)
        if index < len(ids) and str(ids[index]).startswith("person:"):
            return str(ids[index])[7:]
        return ""

    @classmethod
    def _read_csv(cls, path: Path, use_cache: bool = False) -> pd.DataFrame | None:
        try:
            resolved = path.resolve()
            stat = resolved.stat()
            cached = cls._TABLE_CACHE.get(resolved) if use_cache else None
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                return cached[2]
            df = pd.read_csv(resolved, encoding="utf-8", dtype=str, low_memory=False)
            if use_cache:
                cls._TABLE_CACHE[resolved] = (stat.st_mtime_ns, stat.st_size, df)
            return df
        except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError):
            return None

    @classmethod
    def _extract_query_terms(cls, query: str) -> list[str]:
        normalized = re.sub(r"[，。！？、；：,.!?;:]", " ", str(query))
        for word in sorted(cls._QUERY_STOPWORDS, key=len, reverse=True):
            normalized = normalized.replace(word, " ")
        raw_terms = re.findall(r"[A-Za-z0-9+#.-]+|[\u4e00-\u9fff]{2,}", normalized)
        terms: list[str] = []
        for term in raw_terms:
            if term not in terms and len(term.strip()) >= 2:
                terms.append(term.strip())
        return terms

    @staticmethod
    def _row_to_document(row: pd.Series) -> str:
        parts = []
        for column, value in row.items():
            if pd.notna(value) and str(value).strip() and str(value).lower() != "nan":
                parts.append(f"{column}: {value}")
        return "\n".join(parts)


__all__ = ["RAGEngine"]
