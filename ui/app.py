"""Streamlit UI for the HR Multi-Agent Recruitment System."""
from __future__ import annotations
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent


st.set_page_config(
    page_title="HR 多智能体招聘系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize ──

@st.cache_resource
def get_orchestrator():
    return OrchestratorAgent()

orch = get_orchestrator()

# ── Sidebar ──

st.sidebar.title("🤖 HR 多智能体系统")
st.sidebar.markdown("轻量级 Multi-Agent 招聘平台（原生 Tool Calling）")

mode = st.sidebar.radio(
    "选择模式",
    [
        "💬 智能对话",
        "🔍 候选人筛选",
        "📊 深度分析",
        "🎙️ 面试模拟",
        "📝 生成报告",
        "🗂️ 数据探索",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 智能体架构")
st.sidebar.markdown("""
- 🧠 **Orchestrator** — 总控路由
- 📡 **Data Agent** — 数据探索
- 🔎 **Screening Agent** — 筛选匹配
- 📈 **Analysis Agent** — 深度分析
- 🎤 **Interview Agent** — 面试生成
- 📋 **Report Agent** — 报告生成
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 数据概览")
if st.sidebar.button("📊 显示数据集概览"):
    with st.spinner("正在加载数据..."):
        overview = orch.explore_dataset()
        st.sidebar.info(overview[:500] + "..." if len(overview) > 500 else overview)


# ── Main Content ──

st.title("🤖 HR 多智能体招聘系统")
st.caption("基于 OpenAI/DeepSeek Tool Calling + RAG 的多智能体协同招聘平台")

if mode == "💬 智能对话":
    st.header("💬 智能对话")
    st.markdown("可以与系统进行自然语言对话，系统会自动路由到合适的 Agent。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("请输入你的问题（例如：帮我寻找有发明专利的候选人）")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("智能体正在处理..."):
                response = orch.run(prompt)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})


elif mode == "🔍 候选人筛选":
    st.header("🔍 候选人筛选与匹配")
    col1, col2 = st.columns([1, 1])
    with col1:
        person_id = st.text_input("候选人 ID（人员ID）", placeholder="例如：1")
    with col2:
        requirements = st.text_area("岗位要求描述", placeholder="例如：需要博士学历、有发明专利、5年以上管理经验")

    if st.button("开始筛选", type="primary"):
        if person_id:
            with st.spinner("AI 筛选评估中..."):
                prompt = f"请筛选评估候选人 {person_id}。岗位要求: {requirements}"
                result = orch.run(prompt)
            st.markdown("### 筛选结果")
            st.markdown(result)
        else:
            st.warning("请输入候选人 ID")

elif mode == "📊 深度分析":
    st.header("📊 深度能力分析")
    person_id = st.text_input("候选人 ID", placeholder="例如：1")
    context = st.text_area("分析背景（可选）", placeholder="例如：目标岗位为高级技术管理岗")

    if st.button("开始分析", type="primary"):
        if person_id:
            with st.spinner("正在深度分析..."):
                prompt = f"请对候选人 {person_id} 进行深度能力分析。背景: {context}"
                result = orch.run_pipeline("analysis_agent", prompt)
            st.markdown("### 分析报告")
            st.markdown(result)
        else:
            st.warning("请输入候选人 ID")

elif mode == "🎙️ 面试模拟":
    st.header("🎙️ 面试问题生成")
    person_id = st.text_input("候选人 ID", placeholder="例如：1")
    position = st.text_input("目标岗位", placeholder="例如：高级工程师")
    count = st.slider("问题数量", 1, 10, 5)

    if st.button("生成面试问题", type="primary"):
        if person_id:
            with st.spinner("AI 正在生成面试问题..."):
                prompt = f"请为候选人 {person_id} 生成 {count} 个针对性的面试问题。目标岗位: {position}"
                result = orch.run_pipeline("interview_agent", prompt)
            st.markdown("### 面试问题")
            st.markdown(result)
        else:
            st.warning("请输入候选人 ID")

elif mode == "📝 生成报告":
    st.header("📝 招聘报告生成")
    position_name = st.text_input("岗位名称", placeholder="例如：技术总监")
    candidate_ids = st.text_area("候选人 ID 列表（逗号分隔）", placeholder="例如：1, 2, 3, 5")

    if st.button("生成报告", type="primary"):
        if position_name and candidate_ids:
            ids = [c.strip() for c in candidate_ids.split(",") if c.strip()]
            with st.spinner("正在生成招聘报告..."):
                prompt = f"请为岗位【{position_name}】生成完整的招聘报告。候选人列表: {', '.join(ids)}"
                result = orch.run_pipeline("report_agent", prompt)
            st.markdown("### 招聘报告")
            st.markdown(result)
        else:
            st.warning("请填写岗位名称和候选人 ID")

elif mode == "🗂️ 数据探索":
    st.header("🗂️ 数据探索")
    st.markdown("使用自然语言探索 HR 数据集。")

    query = st.text_input("搜索查询", placeholder="例如：有博士学历且有发明专利的候选人")
    if st.button("语义搜索", type="primary"):
        if query:
            with st.spinner("正在语义搜索..."):
                results = orch.search_semantic(query, k=5)
            st.markdown("### 搜索结果")
            for r in results:
                with st.expander(f"👤 {r['姓名']} ({r['人员ID']})"):
                    st.markdown(r["摘要"])

    st.markdown("---")
    st.markdown("### 原始数据查询")
    raw_query = st.text_area(
        "输入自然语言查询",
        placeholder="例如：统计不同学历层次的人数分布",
    )
    if st.button("执行查询"):
        if raw_query:
            with st.spinner("AI 正在查询..."):
                result = orch.run(raw_query)
            st.markdown("### 查询结果")
            st.markdown(result)
