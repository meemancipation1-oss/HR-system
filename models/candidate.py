"""Pydantic data models for candidates, positions, and evaluations."""
#它使用 Pydantic 定义了整个 HR 系统所有的数据模型，保证所有 Agent 都使用统一的数据格式进行交流。
#每一个 BaseModel，对应数据库的一张表格
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime


# ═══════════════════════════════════════
#   Raw Data Models (mapping CSV rows)
# ═══════════════════════════════════════

class Individual(BaseModel):
    """核心人员档案"""
    人员ID: str
    姓名: str
    性别: str = ""
    年龄: int = 0
    民族: str = ""
    学历: str = ""
    政治面貌: str = ""
    军衔等级: str = ""
    职务类别: str = ""
    职务层级: str = ""
    岗位ID: str = ""
    来源方式: str = ""
    出生日期: str = ""
    最高专业名称: str = ""
    本科毕业院校: str = ""
    本科专业: str = ""
    硕士毕业院校: str = ""
    硕士专业: str = ""
    博士毕业院校: str = ""
    博士专业: str = ""
    职业资格: str = ""
    专业技能: str = ""
    擅长领域: str = ""
    管理经验: str = ""
    沟通协调能力: str = ""
    逻辑分析能力: str = ""
    抽象思维能力: str = ""
    人格类型: str = ""
    心理特质: str = ""
    婚恋状况: str = ""
    兴趣偏好: str = ""
    家庭结构: str = ""


class Position(BaseModel):
    """岗位信息"""
    岗位ID: str
    岗位名称: str
    单位ID: str = ""
    岗位类型: str = ""
    职务层级: str = ""
    军衔等级: str = ""
    编制数量: int = 1
    必备条件: str = ""
    禁止条件: str = ""
    可选条件: str = ""


class Employment(BaseModel):
    """任职履历"""
    人员ID: str
    姓名: str = ""
    起始时间: str = ""
    终止时间: str = ""
    单位: str = ""
    职务: str = ""
    职业分类: str = ""
    证人命令号: str = ""
    说明: str = ""


class Evaluation(BaseModel):
    """考核评价"""
    人员ID: str
    姓名: str = ""
    考核时间: str = ""
    年度: str = ""
    考核类别: str = ""
    结果等次: str = ""
    组织单位: str = ""
    总体评价: str = ""
    备注: str = ""


class LanguageAbility(BaseModel):
    """语言能力"""
    人员ID: str
    姓名: str = ""
    认证时间: str = ""
    语种: str = ""
    熟练程度: str = ""
    认证等级: str = ""
    认证成绩: str = ""
    能力说明: str = ""


class MajorProject(BaseModel):
    """重大科研项目"""
    人员ID: str
    姓名: str = ""
    起始时间: str = ""
    终止时间: str = ""
    项目名称: str = ""
    编号: str = ""
    来源: str = ""
    经费: str = ""
    发挥作用层次: str = ""
    作用贡献: str = ""
    备注: str = ""


class PatentInvention(BaseModel):
    """发明专利"""
    人员ID: str
    姓名: str = ""
    申请时间: str = ""
    授权时间: str = ""
    专利类型: str = ""
    专利名称: str = ""
    专利号: str = ""
    排名: str = ""
    简介: str = ""
    作用贡献: str = ""


class Writing(BaseModel):
    """论文著作"""
    人员ID: str
    姓名: str = ""
    刊载时间: str = ""
    文著名称: str = ""
    著述类别: str = ""
    刊载物: str = ""
    排名: str = ""
    简介: str = ""


class TechnicalAward(BaseModel):
    """科技奖励"""
    人员ID: str
    姓名: str = ""
    获奖时间: str = ""
    获奖名称: str = ""
    项目名称: str = ""
    证书号: str = ""
    排名: str = ""
    作用贡献: str = ""
    项目简介: str = ""


class MilitaryExercise(BaseModel):
    """军事演习"""
    人员ID: str
    姓名: str = ""
    起始时间: str = ""
    终止时间: str = ""
    演习类别: str = ""
    组织单位级别: str = ""
    演习名称: str = ""
    担任职务: str = ""
    完成任务情况: str = ""


class Competition(BaseModel):
    """比武竞赛"""
    人员ID: str
    姓名: str = ""
    起始时间: str = ""
    终止时间: str = ""
    比赛名称: str = ""
    比赛类别: str = ""
    组织单位级别: str = ""
    考核比武获奖等级: str = ""
    比赛项目及成绩: str = ""


class SoftwareCopyright(BaseModel):
    """软件著作权"""
    人员ID: str
    姓名: str = ""
    登记时间: str = ""
    著作权名称: str = ""
    简称: str = ""
    登记号: str = ""
    排名: str = ""


# ═══════════════════════════════════════
#   Composite / Application Models
# ═══════════════════════════════════════

class CandidateProfile(BaseModel):
    """Composite candidate profile assembled from all raw tables."""
    #将所有的组合起来
    individual: Individual
    employment_history: list[Employment] = []
    evaluations: list[Evaluation] = []
    languages: list[LanguageAbility] = []
    projects: list[MajorProject] = []
    patents: list[PatentInvention] = []
    writings: list[Writing] = []
    technical_awards: list[TechnicalAward] = []
    exercises: list[MilitaryExercise] = []
    competitions: list[Competition] = []
    software_copyrights: list[SoftwareCopyright] = []


class PositionMatch(BaseModel):
    """职位匹配结果"""
    岗位名称: str
    岗位ID: str
    匹配分数: float = Field(ge=0, le=100)
    匹配理由: str = ""


class CandidateScore(BaseModel):
    """候选人评分"""
    人员ID: str
    姓名: str
    总分: float = Field(ge=0, le=100)
    职位匹配: list[PositionMatch] = []
    技能评分: float = Field(ge=0, le=100)
    经验评分: float = Field(ge=0, le=100)
    教育评分: float = Field(ge=0, le=100)
    综合评价: str = ""


class InterviewQuestion(BaseModel):
    """面试问题"""
    问题: str
    考察点: str
    难度: str = "中等"  # 简单 / 中等 / 困难


class InterviewSession(BaseModel):
    """面试会话"""
    候选人: CandidateProfile
    问题列表: list[InterviewQuestion] = []
    问答记录: list[dict] = []
    总体评价: str = ""
    建议录用: bool = False


class RecruitmentReport(BaseModel):
    """招聘报告"""
    职位名称: str
    候选人总数: int
    推荐候选人: list[CandidateScore]
    面试安排: list[InterviewSession] = []
    ai_分析总结: str = ""
    生成时间: datetime = Field(default_factory=datetime.now)
