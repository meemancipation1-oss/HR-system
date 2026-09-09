"""FastAPI routes for the HR Agent system."""
from __future__ import annotations
import json
import time
from threading import RLock
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from exercise.pydanticEX import BaseModel

from agents.orchestrator_agent import OrchestratorAgent, TaskType
from agents.screening_agent import ScreeningOutput
from config.settings import SESSION_DB_PATH, LOG_PATH
from core.observability import configure_logging, log_event, trace_context
from core.runtime import RuntimeFailure
from core.session_store import SessionStore

app = FastAPI(
    title="HR Multi-Agent Recruitment System",
    description="Native Tool Calling Multi-Agent Recruitment System API",
    version="1.0.0",
)

# ── Serve static frontend ──
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    def root():
        return {"service": "HR Multi-Agent Recruitment System", "version": "1.0.0"}


_orchestrators: dict[str, OrchestratorAgent] = {}
_orchestrator_lock = RLock()
_session_store = SessionStore(SESSION_DB_PATH)
_logger = configure_logging(LOG_PATH)


def get_orch(session_id: str) -> OrchestratorAgent:
    """Get an isolated orchestrator and restore its durable conversation."""
    with _orchestrator_lock:
        if session_id not in _orchestrators:
            _orchestrators[session_id] = OrchestratorAgent(
                messages=_session_store.load_messages(session_id)
            )
        return _orchestrators[session_id]


def _context(session_id: str | None) -> tuple[str, str, OrchestratorAgent]:
    session_id = session_id or uuid4().hex
    request_id = uuid4().hex
    return request_id, session_id, get_orch(session_id)


def _save_session(session_id: str, orch: OrchestratorAgent) -> None:
    _session_store.save_messages(session_id, orch.get_memory_snapshot())


def _failure(exc: Exception) -> RuntimeFailure:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return RuntimeFailure(status="incomplete", error_code="llm_timeout",
                              message="请求超时，请稍后重试。", retryable=True,
                              details={"error_type": type(exc).__name__})
    return RuntimeFailure(status="error", error_code="internal_error",
                          message="请求处理失败。", retryable=False,
                          details={"error_type": type(exc).__name__})


# ── Request / Response models ──

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class TaskRequest(BaseModel):
    task_type: str = "auto"
    input: str
    session_id: str | None = None


class ScreeningRequest(BaseModel):
    person_id: str
    requirements: str = ""
    position_id: str | None = None
    session_id: str | None = None


class ReportRequest(BaseModel):
    position_name: str
    candidate_ids: list[str]
    session_id: str | None = None


class ReviewRequest(BaseModel):
    candidate_id: str
    status: str
    note: str = ""


# ── Endpoints ──

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        _session_store.load_messages("__ready__")
        return {"status": "ready", "session_store": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_failure(exc).model_dump(mode="json"))


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return {"session_id": session_id,
            "messages": _session_store.load_messages(session_id),
            "task_state": _session_store.load_task_state(session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    _session_store.delete(session_id)
    with _orchestrator_lock:
        _orchestrators.pop(session_id, None)
    return {"deleted": True, "session_id": session_id}


@app.post("/sessions/{session_id}/reviews")
def review_candidate(session_id: str, req: ReviewRequest):
    if req.status not in {"approved", "rejected", "pending"}:
        raise HTTPException(status_code=422, detail="status must be approved, rejected, or pending")
    state = _session_store.load_task_state(session_id)
    reviews = state.setdefault("reviews", {})
    reviews[req.candidate_id] = {"status": req.status, "note": req.note, "updated_at": time.time()}
    _session_store.save_task_state(session_id, state)
    return {"session_id": session_id, "candidate_id": req.candidate_id, "review": reviews[req.candidate_id]}


@app.post("/chat")
def chat(req: ChatRequest):
    """General-purpose chat with the orchestrator."""
    request_id, session_id, orch = _context(req.session_id)
    started = time.time()
    try:
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/chat"):
            log_event(_logger, "request_started")
            result = orch.run(req.message)
        _save_session(session_id, orch)
        log_event(_logger, "request_finished", request_id=request_id, session_id=session_id,
                  endpoint="/chat", latency_ms=round((time.time() - started) * 1000, 1), success=True)
        return {"response": result, "request_id": request_id, "session_id": session_id}
    except Exception as e:
        log_event(_logger, "request_failed", request_id=request_id, session_id=session_id,
                  endpoint="/chat", error_type=type(e).__name__, success=False)
        raise HTTPException(status_code=500, detail=_failure(e).model_dump(mode="json"))


@app.post("/chat/perf")
def chat_with_perf(req: ChatRequest):
    """Chat with performance metrics included."""
    request_id, session_id, orch = _context(req.session_id)
    try:
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/chat/perf"):
            log_event(_logger, "request_started")
            result, metrics, agent_name = orch.run_with_metrics(req.message)
        _save_session(session_id, orch)
        metrics.update({"request_id": request_id, "session_id": session_id})
        return {"response": result, "metrics": metrics, "agent": agent_name,
                "request_id": request_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_failure(e).model_dump(mode="json"))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Return one planned result over SSE for the browser chat UI.

    Natural-language chat is deliberately routed through ``run_with_metrics``.
    The explicit ``/search`` endpoint remains available for callers that need
    direct retrieval rather than intent planning.
    """
    request_id, session_id, orch = _context(req.session_id)

    async def planned_stream():
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/chat/stream"):
            log_event(_logger, "request_started", route="task_planner")
            result, metrics, _ = orch.run_with_metrics(req.message)
            _save_session(session_id, orch)
            log_event(_logger, "request_finished", route="task_planner", success=True,
                      latency_ms=metrics["elapsed_ms"])
            payload = {
                "type": "content",
                "text": result,
                "request_id": request_id,
                "session_id": session_id,
                "metrics": metrics,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(planned_stream(), media_type="text/event-stream")


@app.post("/task")
def execute_task(req: TaskRequest):
    """Execute a specific task type."""
    request_id, session_id, orch = _context(req.session_id)
    try:
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/task"):
            log_event(_logger, "request_started", task_type=req.task_type)
            task_enum = TaskType(req.task_type) if req.task_type != "auto" else None
            if task_enum:
                result = orch.run_pipeline(task_enum, req.input)
            else:
                result = orch.run(req.input)
        _save_session(session_id, orch)
        log_event(_logger, "request_finished", request_id=request_id, session_id=session_id,
                  endpoint="/task", success=True)
        return {"response": result, "request_id": request_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_failure(e).model_dump(mode="json"))


@app.post("/screen")
def screen(req: ScreeningRequest):
    """Screen a candidate against requirements."""
    request_id, session_id, orch = _context(req.session_id)
    try:
        # Screening is request-scoped: never let one candidate's evidence leak
        # into another candidate's structured decision.
        from agents.screening_agent import ScreeningAgent
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/screen"):
            log_event(_logger, "request_started", candidate_id=req.person_id)
            agent = ScreeningAgent()
            raw = agent.screen_candidate(req.person_id, position_id=req.position_id, requirements=req.requirements)
        output = raw if isinstance(raw, ScreeningOutput) else ScreeningOutput(**json.loads(raw))
        if output.数据矛盾 or not output.证据:
            output.需要人工复核 = True
            output.决策 = "review"
        orch._memory.add_user_message(
            f"筛选候选人 {req.person_id}，岗位要求: {req.requirements}"
        )
        orch._memory.add_ai_message(output.model_dump_json(ensure_ascii=False))
        _save_session(session_id, orch)
        state = _session_store.load_task_state(session_id)
        state["current_job"] = {"position_id": req.position_id, "requirements": req.requirements}
        screenings = state.setdefault("screenings", {})
        screenings[req.person_id] = output.model_dump(mode="json")
        state.setdefault("selected_candidates", [])
        if req.person_id not in state["selected_candidates"]:
            state["selected_candidates"].append(req.person_id)
        _session_store.save_task_state(session_id, state)
        log_event(_logger, "request_finished", request_id=request_id, session_id=session_id,
                  endpoint="/screen", candidate_id=req.person_id, decision=output.决策,
                  human_review=output.需要人工复核, success=True)
        return {"screening": output.model_dump(mode="json"),
                "request_id": request_id, "session_id": session_id}
    except Exception as e:
        return {"screening": None, "request_id": request_id, "session_id": session_id,
                "failure": _failure(e).model_dump(mode="json")}


@app.post("/report")
def generate_report(req: ReportRequest):
    """Generate recruitment report for position and candidates."""
    request_id, session_id, orch = _context(req.session_id)
    try:
        with trace_context(request_id=request_id, session_id=session_id, endpoint="/report"):
            log_event(_logger, "request_started", candidate_count=len(req.candidate_ids))
            prompt = (
                f"请为岗位【{req.position_name}】生成招聘报告。"
                f"候选人列表: {', '.join(req.candidate_ids)}"
            )
            result = orch.run(prompt)
        _save_session(session_id, orch)
        state = _session_store.load_task_state(session_id)
        state["latest_report"] = {"position_name": req.position_name,
                                  "candidate_ids": req.candidate_ids, "result": result,
                                  "updated_at": time.time()}
        _session_store.save_task_state(session_id, state)
        return {"response": result, "request_id": request_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_failure(e).model_dump(mode="json"))


@app.get("/explore")
def explore_dataset():
    """Get dataset overview."""
    _, _, orch = _context(None)
    try:
        result = orch.explore_dataset()
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def semantic_search(query: str, top_k: int = 5):
    """Semantic search over candidates (direct RAG, zero LLM calls)."""
    _, _, orch = _context(None)
    try:
        results = orch.search_semantic(query, k=top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
