# 🤖 HR 多智能体招聘系统

> **秋招简历项目 — 轻量级 Multi-Agent 系统（原生 Tool Calling）**
> 将 10 万+ 条军事 HR 数据转化为智能招聘决策引擎
> 零框架依赖，手写 Tool Calling 循环，面试可讲透每一行代码

---

## 🎯 项目定位

**面向 AI Agent / LLM 应用开发岗位的简历项目。**

展示的核心能力：
1. **Multi-Agent 架构设计** — 6 个专业 Agent + 智能路由
2. **原生 Tool Calling 循环** — 手写 OpenAI/DeepSeek 函数调用
3. **RAG 检索增强生成** — ChromaDB + 本地语义搜索
4. **结构化输出 (Forced Tool Calling)** — 适配 DeepSeek 的受限能力
5. **工程化落地能力** — FastAPI + Streamlit + CLI 三端部署
6. **零框架依赖** — 不依赖 LangChain/LangGraph/CrewAI，面试不怕追问细节

---

## 🏗️ 架构总览

```
用户输入 ──→ 🧠 Orchestrator Agent ──→ LLM 语义分类
                  │
        ┌─────────┼──────────┬──────────┐
        ▼         ▼          ▼          ▼
   📡 Data    🔎 Screening  📈 Analysis  🎤 Interview
   Agent      Agent         Agent        Agent
        │         │          │          │
        └─────────┴──────────┴──────────┘
                  ▼
           📋 Report Agent
                  │
                  ▼
         ┌────────┴────────┐
         ▼                  ▼
    CLI / API / UI     📊 结构化报告
```

### 核心 Agent 循环（手写）

```python
for iteration in range(MAX_TURNS):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_schema,     # 自描述工具
        tool_choice="auto",
    )
    if no tool_calls:
        return response.content  # 最终回复
    for tool_call in tool_calls:
        result = execute(tool_call)  # 纯函数执行
        messages.append(tool_result) # 追加结果
```

### 技术选型对比

| 层次 | 本方案 | 常规方案 | 选择理由 |
|------|--------|----------|----------|
| **Agent 框架** | 手写 Tool Calling | LangGraph / CrewAI | 零抽象泄漏，面试可讲透 |
| **LLM 调用** | `openai` SDK 原生 | `langchain-openai` | 1 行依赖 vs 10+ 行依赖 |
| **结构化输出** | Forced Tool Calling | `with_structured_output` | DeepSeek 不支持原生 JSON mode |
| **向量检索** | `chromadb` 原生 | `langchain-chroma` | 去 LC 文档包装，直查 dict |
| **数据模型** | Pydantic v2 | — | 验证 + schema 导出 + 序列化 |
| **接口层** | FastAPI | — | 标准选择 |
| **前端** | Streamlit | — | 标准选择 |

---

## 🤖 六大智能体详解

### 1. 🧠 Orchestrator Agent（总控编排）
- LLM 语义分类用户意图 → 路由到对应子 Agent
- 支持自动分类和手动指定 pipeline
- 纯 Python 路由，无需状态机

### 2. 📡 Data Agent（数据探索）
- 查询人员档案、组织单位、岗位信息
- 执行多表 RAG 语义搜索（所有含 `人员ID` 的 CSV 按人员聚合后写入 ChromaDB）
- 支持任职、语言、专利、论文、项目、军事经历等关联表条件的候选人召回
- 对任意 HR CSV 提供结构化关键词检索；数据统计与概览

### 3. 🔎 Screening Agent（筛选匹配）
- 6 维度评估：教育、技能、经验、能力、成果、特殊条件
- **结构化评分输出** — 使用 Forced Tool Calling 保证 JSON 格式
- 适配 DeepSeek 无法使用 `response_format=json_object` 的限制

### 4. 📈 Analysis Agent（深度分析）
- 能力画像矩阵
- 发展潜力预测
- 劣势与改进建议

### 5. 🎤 Interview Agent（面试生成）
- 4 类问题：技术、情景、行为、综合
- 个性化生成（基于候选人背景）
- 回答质量评估

### 6. 📋 Report Agent（报告生成）
- 多候选人对比排名
- 结构化招聘报告
- 决策建议输出

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 依赖：`pip install -r requirements.txt`

### 配置

```bash
# 复制并修改 .env
cp .env.example .env
# 填入你的 API Key（支持 DeepSeek / OpenAI 等兼容接口）
# OPENAI_API_KEY=sk-xxx
```

### 运行方式

```bash
# 1. 构建向量索引
python main.py ingest

# 1.5 (可选) LLM-as-Judge 评估简历筛选准确率
python main.py evaluate --cases 5     # 每个场景 5 个案例, 完整流程
python main.py evaluate --offline    # 只构建黄金集, 不调用 LLM

# 2. 启动（三选一）
python main.py demo           # CLI 交互演示
python main.py api             # FastAPI 服务 → http://localhost:8000/docs
streamlit run ui/app.py        # Web 界面
```

### 演示案例

| 功能 | 输入示例 | 输出 |
|------|---------|------|
| 🔍 语义搜索 | "寻找有博士学历且有发明专利的候选人" | 5 位匹配候选人列表 |
| 📊 能力分析 | "对人员 1 进行深度面试分析" | 完整能力画像报告 |
| 🎙️ 面试生成 | "为人员 1 生成 5 个面试题" | 5 个个性化问题+考察点 |
| 📝 招聘报告 | "为技术总监岗生成招聘报告" | 多候选人对比+推荐 |
| 💬 自由对话 | "帮我统计硕士以上学历的人数" | AI 自动查询并统计 |

---

## 🧪 LLM-as-Judge 评估模块（简历筛选准确率）

评估 Screening Agent 的简历筛选质量，采用业界标准的 **LLM-as-Judge** 范式：

```
规则引擎(ground truth) ──┐
                        ├──► Judge LLM ──► 指标报告
ScreeningAgent(被测系统) ─┘        (准确率/F1/MAE)
```

### 核心流程
1. **黄金测试集** — 用确定性规则（学历/专利/论文/年龄/技能等）从 CSV 构建真实标签，正负样本均衡、可复现
2. **被测系统** — ScreeningAgent 对同样 候选人+岗位要求 执行筛选，输出结构化评分
3. **LLM-as-Judge** — 独立大模型参照真实标签判定 Agent 结论（正确/部分正确/错误 + 一致程度 + 误判原因）
4. **指标汇总** — 准确率 / 精确率 / 召回率 / F1 / 评分MAE / Judge 一致率，按场景拆分 + 全量明细 JSON

### 设计亮点
- **参照真实标签的 Judge**（reference-guided）：Judge 同时看到规则依据与 Agent 输出，可判断"结论是否正确"而非"答得是否流畅"
- **独立 Judge 模型**：支持 `JUDGE_MODEL` 环境变量指定更强模型，避免"自评自"的偏好偏差
- **真实发现问题**：评估曾捕获 ScreeningAgent 调取错人档案的 bug（全字段模糊搜索误命中），修复后误判率显著下降

### 输出物
- `data/eval_cases.json` — 黄金测试集（可复现）
- `data/eval_results.json` — 完整评估报告（指标 + 每个案例的 Judge 判定）

---

## 🔧 架构决策记录（面试向）

### 为什么不用 LangChain/LangGraph？

**核心评估：我的场景不需要框架。**

LangGraph 适合有复杂状态流转的场景（多个 Agent 之间有循环、分支、条件跳转）。我的系统中，Agent 之间是**树形协作**关系：Orchestrator 分类 → 单个子 Agent 执行 → 返回结果。这是一个有向无环图，不需要 StateGraph。

**手写 Tool Calling 带来的优势：**
1. **完全控制** — 错误重试、并行执行、Token 管理，每个细节都可控
2. **依赖极简** — `requirements.txt` 从 15 个依赖降到 7 个
3. **可解释性** — 面试时能清晰讲清楚每一行代码的作用
4. **调试友好** — 没有框架抽象层，报错直接定位

### 为什么使用 Forced Tool Calling 实现结构化输出？

DeepSeek API **不支持** OpenAI 的 `response_format={"type": "json_object"}`。解决方案：定义一个只有一个函数的 tool，用 `tool_choice={"type": "function", "function": {"name": "output_result"}}` 强制模型调用，从而获得结构化 JSON。这个模式也适用于其他非 OpenAI 的兼容 API。

### 为什么用本地 sentence-transformers 做 Embedding？

DeepSeek 没有公开的 Embedding API。使用本地模型（`all-MiniLM-L6-v2`）的优势：
- **零延迟** — 无需网络请求
- **离线可用** — 部署不受网络限制
- **免费** — 无 API 调用费用

### 多表 RAG 的召回策略

系统采用“候选人级向量召回 + 关联表结构化检索”的混合方案。索引构建时扫描所有包含 `人员ID` 的 CSV，将同一人的主档案、任职、语言、成果和经历记录聚合为一条候选人文档，并在文档中保留来源表和行号。没有 `人员ID` 的岗位、组织表通过 `search_table` 工具查询，不与个人画像混合。

学历、语种、认证等级、任职单位、时间等精确条件使用 `search_candidates_all_tables` 或 `search_table` 做字段/文本过滤；“适合技术管理”“项目管理经验丰富”等描述性请求使用 ChromaDB 向量召回。最终结果统一返回 `人员ID`，再进入筛选、分析和报告流水线。

索引结构升级后需要重建一次：

```bash
python scripts/ingest_data.py
```

### Tool 设计原则

每个 Tool 由一个纯执行函数 + 一个 Pydantic 输入模型组成，可以自动导出为 OpenAI Function Calling 格式。这种设计：
- **类型安全** — 输入在 Python 侧由 Pydantic 校验
- **自描述** — schema 自动映射为 tool 参数描述
- **可测试** — 纯函数，无需 mock LLM

---

## 📂 项目结构

```
hr-agent/
├── main.py                    # 统一入口
├── agents/                    # 🤖 Agent 实现
│   ├── orchestrator_agent.py  #   总控路由（LLM 分类）
│   ├── base_agent.py          #   基类（Tool Calling 循环）
│   ├── data_agent.py          #   数据探索
│   ├── screening_agent.py     #   筛选匹配（结构化输出）
│   ├── analysis_agent.py      #   深度分析
│   ├── interview_agent.py     #   面试生成
│   └── report_agent.py        #   报告生成
├── core/                      # ⚙️ 核心框架
│   ├── llm.py                 #   LLM 接口（原生 OpenAI SDK）
│   ├── tools.py               #   工具集（Pydantic + Function Calling）
│   ├── rag.py                 #   ChromaDB RAG（无 LC 包装）
│   ├── memory.py              #   对话记忆
│   └── evaluator.py           #   LLM-as-Judge 评估模块
├── models/                    # 📦 数据模型
│   └── candidate.py           #   20+ Pydantic 模型
├── api/routes.py              # 🌐 FastAPI 接口
├── ui/app.py                  # 🖥️ Streamlit UI
└── config/settings.py         # ⚙️ 全局配置
```

---

## 📊 简历亮点提炼

### 技术层面
```
✅ 独立设计并实现了一个生产级 Multi-Agent 系统，零框架依赖
✅ 手写 Tool Calling 循环，理解 OpenAI/DeepSeek 函数调用本质
✅ 使用 Forced Tool Calling 适配 DeepSeek 的结构化输出限制
✅ 实现 RAG 检索增强（ChromaDB + sentence-transformers）
✅ 自建 LLM-as-Judge 评估体系，量化简历筛选准确率（准确率/F1/MAE）
✅ Pydantic 结构化输出 + 自动 schema 导出
✅ FastAPI + Streamlit + CLI 三端部署
```

### 架构设计
```
✅ 微服务化的 Agent 架构，职责单一
✅ 智能路由 + LLM 语义分类，自动分配任务
✅ Tool 自描述设计，纯函数可测试
✅ 错误隔离（单 Agent 失败不影响整体）
```

### 业务价值
```
✅ 将传统 HR 数据转化为智能招聘引擎
✅ 实现从数据探索 → 筛选 → 分析 → 面试 → 报告的全流程自动化
✅ AI Agent 理解业务语义，而非简单关键词匹配
```

---

## 🔗 相关资源

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs)
- [ChromaDB 向量数据库](https://www.trychroma.com/)
- [Pydantic v2 文档](https://docs.pydantic.dev/latest/)




