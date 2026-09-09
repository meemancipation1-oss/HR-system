"""让 LLM 能够通过 Function Calling 调用 Python 函数，从数据库（CSV）里查询数据。"""

from __future__ import annotations
import json
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import HR_DATA_DIR


# ── In-memory data cache ──
_tables: dict[str, pd.DataFrame] = {}


def _load_table(name: str) -> pd.DataFrame:
    """Lazy-load a CSV table (cached).缓存之后不用再读取"""
    if name not in _tables:
        path = HR_DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Table {name} not found at {path}")
        _tables[name] = pd.read_csv(
            path,
            encoding="utf-8",
            dtype=str,
            low_memory=False,
        )
    return _tables[name]


# ═══════════════════════════════════════
#  Tool input schemas
# ═══════════════════════════════════════

class SearchIndividualInput(BaseModel):
    keyword: str = Field(description="搜索关键词：姓名、人员ID 或其他字段值")


class GetIndividualDetailInput(BaseModel):
    person_id: str = Field(description="人员ID，例如 '1'")


class SearchPositionsInput(BaseModel):
    keyword: str = Field(description="搜索关键词：岗位名称或ID")


class GetPositionDetailInput(BaseModel):
    position_id: str = Field(description="岗位ID")


class GetEmploymentHistoryInput(BaseModel):
    person_id: str = Field(description="人员ID")


class GetEvaluationsInput(BaseModel):
    person_id: str = Field(description="人员ID")


class GetTechnicalAchievementsInput(BaseModel):
    person_id: str = Field(description="人员ID")


class GetSkillsAndAbilitiesInput(BaseModel):
    person_id: str = Field(description="人员ID")


class GetMilitaryExperienceInput(BaseModel):
    person_id: str = Field(description="人员ID")


class GetPersonRecordsInput(BaseModel):
    person_id: str = Field(description="人员ID")
    tables: list[str] = Field(
        default_factory=list,
        description="可选的表名列表；为空时查询所有含人员ID的人员表",
    )


class SearchCandidatesAllTablesInput(BaseModel):
    keyword: str = Field(description="跨人员关联表检索条件，例如英语、某单位、发明专利")
    k: int = Field(default=20, ge=1, le=100, description="最多返回候选人数")


class SearchTableInput(BaseModel):
    table_name: str = Field(description="CSV 表名，不含 .csv，例如 LanguageAbility")
    keyword: str = Field(description="要匹配的字段值或关键词")
    k: int = Field(default=20, ge=1, le=100, description="最多返回记录数")


# ═══════════════════════════════════════
#  Tool implementations
# ═══════════════════════════════════════

def search_individual(input: SearchIndividualInput) -> str:
    """Search for individuals by name, ID or keyword."""
    df = _load_table("SyntheticIndividuals")
    keyword = input.keyword.strip()
    # 精确匹配优先: 人员ID / 姓名 完全一致, 避免模糊匹配把无关人员带出
    exact = df[(df["人员ID"] == keyword) | (df["姓名"] == keyword)]
    if not exact.empty:
        return str(exact.head(10).to_dict(orient="records"))
    # 回退: 全字段模糊匹配
    mask = df.apply(
        lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(),
        axis=1,
    )
    results = df[mask].head(10).to_dict(orient="records")
    return str(results) if results else "未找到匹配人员"


def get_individual_detail(input: GetIndividualDetailInput) -> str:
    """Get full profile of a single individual by 人员ID."""
    df = _load_table("SyntheticIndividuals")
    row = df[df["人员ID"] == input.person_id]
    if row.empty:
        return f"未找到人员ID={input.person_id}"
    return str(row.iloc[0].to_dict())


def search_positions(input: SearchPositionsInput) -> str:
    """Search for positions by name or ID."""
    df = _load_table("SyntheticPositions")
    keyword = input.keyword
    mask = df.apply(
        lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(),
        axis=1,
    )
    results = df[mask].head(10).to_dict(orient="records")
    return str(results) if results else "未找到匹配岗位"


def get_position_detail(input: GetPositionDetailInput) -> str:
    """Get full information for a position."""
    df = _load_table("SyntheticPositions")
    row = df[df["岗位ID"] == input.position_id]
    if row.empty:
        return f"未找到岗位ID={input.position_id}"
    return str(row.iloc[0].to_dict())


def get_employment_history(input: GetEmploymentHistoryInput) -> str:
    """Get employment / 任职 history for a person."""
    df = _load_table("employment")
    rows = df[df["人员ID"] == input.person_id]
    if rows.empty:
        return f"未找到任职记录: {input.person_id}"
    return str(rows.to_dict(orient="records"))


def get_evaluations(input: GetEvaluationsInput) -> str:
    """Get evaluation / 考核 records for a person."""
    df = _load_table("EvaluationInfo")
    rows = df[df["人员ID"] == input.person_id]
    if rows.empty:
        return "未找到考核记录"
    return str(rows.to_dict(orient="records"))


def get_technical_achievements(input: GetTechnicalAchievementsInput) -> str:
    """Get all technical achievements: patents, papers, awards, software copyrights, projects."""
    out = {}
    for tbl, label in [
        ("PatentInvention", "发明专利"),
        ("Writing", "论文著作"),
        ("TechnicalAward", "科技奖励"),
        ("SoftwareCopyright", "软件著作权"),
        ("MajorSciProject", "科研项目"),
    ]:
        df = _load_table(tbl)
        rows = df[df["人员ID"] == input.person_id]
        out[label] = rows.to_dict(orient="records") if not rows.empty else []
    return str(out)


def get_skills_and_abilities(input: GetSkillsAndAbilitiesInput) -> str:
    """Get language abilities, professional skills, and psychological traits."""
    out: dict = {}
    # Language
    df_lang = _load_table("LanguageAbility")
    lang_rows = df_lang[df_lang["人员ID"] == input.person_id]
    out["语言能力"] = lang_rows.to_dict(orient="records") if not lang_rows.empty else []

    # Individual record has embedded skill fields
    df_ind = _load_table("SyntheticIndividuals")
    ind_row = df_ind[df_ind["人员ID"] == input.person_id]
    if not ind_row.empty:
        r = ind_row.iloc[0]
        for field in [
            "专业技能", "擅长领域", "管理经验", "沟通协调能力", "逻辑分析能力",
            "抽象思维能力", "人格类型", "心理特质", "职业资格",
        ]:
            if field in r and pd.notna(r[field]):
                out[field] = r[field]
    return str(out)


def get_military_experience(input: GetMilitaryExperienceInput) -> str:
    """Get military experience including exercises, competitions, war/non-war ops."""
    out = {}
    for tbl, label in [
        ("MilitaryExercise", "军事演习"),
        ("MilitaryCompetitionExperience", "比武竞赛"),
        ("War", "作战经历"),
        ("NonWar", "非战争军事行动"),
    ]:
        df = _load_table(tbl)
        rows = df[df["人员ID"] == input.person_id]
        out[label] = rows.to_dict(orient="records") if not rows.empty else []
    return str(out)


def get_person_records(input: GetPersonRecordsInput) -> str:
    """Get all records for one person across every related HR CSV table."""
    import json

    table_names = input.tables or [path.stem for path in HR_DATA_DIR.glob("*.csv")]
    records: dict[str, list[dict]] = {}
    for table_name in sorted(set(table_names)):
        path = HR_DATA_DIR / f"{table_name}.csv"
        if not path.exists():
            raise ValueError(f"未知数据表: {table_name}")
        try:
            if "人员ID" not in pd.read_csv(path, encoding="utf-8", nrows=0).columns:
                continue
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ValueError(f"无法读取数据表: {table_name}") from exc
        df = _load_table(table_name)
        rows = df[df["人员ID"].astype(str) == str(input.person_id)]
        if not rows.empty:
            records[table_name] = [
                {str(key): ("" if pd.isna(value) else str(value)) for key, value in row.items()}
                for _, row in rows.iterrows()
            ]
    return json.dumps(records, ensure_ascii=False)


def search_candidates_all_tables(input: SearchCandidatesAllTablesInput) -> str:
    """Recall candidate IDs by searching every CSV that contains 人员ID."""
    from core.rag import RAGEngine

    engine = RAGEngine()
    results = engine.search_candidates_in_all_tables(input.keyword, k=input.k)
    return json.dumps(results, ensure_ascii=False)


def search_table(input: SearchTableInput) -> str:
    """Search any HR CSV table by a field value or free-text keyword."""
    from core.rag import RAGEngine

    engine = RAGEngine()
    results = engine.search_table(input.table_name, input.keyword, k=input.k)
    return json.dumps(results, ensure_ascii=False)


# ═══════════════════════════════════════
#  Tool registry
# ═══════════════════════════════════════

class ToolDef:
    """A tool definition: implementation function + input schema."""

    def __init__(self, fn: callable, input_model: type[BaseModel]):
        self.fn = fn
        self.input_model = input_model
        self.name = fn.__name__
        self.description = fn.__doc__ or ""

    def to_openai_tool(self) -> dict:
        """Export this tool as an OpenAI / DeepSeek function-calling tool dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

    def execute(self, arguments: str) -> str:
        """Execute the tool with a JSON string of arguments."""
        import json
        parsed = json.loads(arguments)
        validated = self.input_model(**parsed)
        return self.fn(validated)


# ── All tools ──  #整个系统的工具注册中心
ALL_TOOLS: list[ToolDef] = [
    ToolDef(search_individual, SearchIndividualInput),
    ToolDef(get_individual_detail, GetIndividualDetailInput),
    ToolDef(search_positions, SearchPositionsInput),
    ToolDef(get_position_detail, GetPositionDetailInput),
    ToolDef(get_employment_history, GetEmploymentHistoryInput),
    ToolDef(get_evaluations, GetEvaluationsInput),
    ToolDef(get_technical_achievements, GetTechnicalAchievementsInput),
    ToolDef(get_skills_and_abilities, GetSkillsAndAbilitiesInput),
    ToolDef(get_military_experience, GetMilitaryExperienceInput),
    ToolDef(get_person_records, GetPersonRecordsInput),
    ToolDef(search_candidates_all_tables, SearchCandidatesAllTablesInput),
    ToolDef(search_table, SearchTableInput),
]


def get_all_tools_schema() -> list[dict]:
    """Get the OpenAI-compatible tool schema for all tools."""
    return [t.to_openai_tool() for t in ALL_TOOLS]


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool by name with JSON arguments. Raises ValueError if not found."""
    #这是模型真正调用工具时的统一入口。
    for t in ALL_TOOLS:
        if t.name == name:
            return t.execute(arguments)
    raise ValueError(f"未知工具: {name}")


__all__ = [
    "ALL_TOOLS",
    "get_all_tools_schema",
    "execute_tool",
    "ToolDef",
]

