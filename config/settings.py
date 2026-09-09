"""Global configuration for HR Agent system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── LLM ──
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL     = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
LLM_MODEL           = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEMPERATURE         = float(os.getenv("TEMPERATURE", "0.1"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))

# ── Proxy ──
# Automatically apply proxy from env so requests go through the proxy
HTTP_PROXY          = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""
HTTPS_PROXY         = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or ""
if HTTP_PROXY:
    os.environ.setdefault("HTTP_PROXY", HTTP_PROXY)
    os.environ.setdefault("http_proxy", HTTP_PROXY)
if HTTPS_PROXY:
    os.environ.setdefault("HTTPS_PROXY", HTTPS_PROXY)
    os.environ.setdefault("https_proxy", HTTPS_PROXY)

# ── Data paths ──
HR_DATA_DIR         = Path(os.getenv("HR_DATA_DIR", PROJECT_ROOT.parent / "SynData_1w"))
VECTOR_DB_DIR       = Path(os.getenv("VECTOR_DB_DIR", PROJECT_ROOT / "data" / "chroma_db"))
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DB_PATH     = Path(os.getenv("SESSION_DB_PATH", PROJECT_ROOT / "data" / "sessions.sqlite3"))
LOG_PATH            = Path(os.getenv("LOG_PATH", PROJECT_ROOT / "data" / "events.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Agent config ──
AGENT_CONFIG = {
    "max_iterations": 10,
    "early_stopping_method": "generate",
    "verbose": True,
}

# ── Tables that can be used ──
AVAILABLE_TABLES = {
    "SyntheticIndividuals":     "核心人员档案（含人口学、教育、军事履历、技能、心理等）",
    "SyntheticPositions":       "岗位信息（岗位名称、单位、层级、编制、条件）",
    "SyntheticOrganizations":   "组织单位信息",
    "employment":               "任职履历",
    "EvaluationInfo":           "考核评价信息",
    "LanguageAbility":          "语言能力（语种、熟练程度、认证等级）",
    "MajorSciProject":          "重大科研项目参与经历",
    "PatentInvention":          "发明专利信息",
    "Writing":                  "论文著作发表记录",
    "SoftwareCopyright":        "软件著作权",
    "StandardSpecification":    "标准规范参与",
    "TechnicalAward":           "科技获奖记录",
    "TechnicalTalent":          "技术人才认定",
    "TechnicalTalentAward":     "技术人才表彰",
    "MilitaryExercise":         "军事演习经历",
    "MilitaryCompetitionExperience": "军事比武竞赛经历",
    "RocketForceExperience":    "火箭军经历",
    "CaptainExperience":        "舰艇长经历",
    "FlightExperience":         "飞行经历",
    "WeaponEquipment":          "武器装备操作能力",
    "AwardPunishment":          "奖惩记录",
    "MedalInformation":         "勋章信息",
    "NonWar":                   "非战争军事行动",
    "War":                      "作战经历",
    "GroupParticipation":       "团体/组织参与",
    "ConfidentialStatus":       "涉密状态",
    "PhysicalExamInfo":         "体检信息",
    "PersonalDynamicStatus":    "人员在位动态",
}
