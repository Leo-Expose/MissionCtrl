"""FastAPI application for MissionCtrl Environment.

Uses a persistent singleton MissionCtrlEnvironment so that state is preserved
across /reset and /step calls.
"""

import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from server.environment import MissionCtrlEnvironment
from models import BuildMetadata, HeartbeatResponse, LogsSummaryResponse, RequestLogEntry

# ---------------------------------------------------------------------------
# Logging — suppress noisy poll endpoints
# ---------------------------------------------------------------------------
class _PollFilter(logging.Filter):
    _SUPPRESSED = ("/state", "/history", "/results", "/dashboard", "/health", "/favicon", "/apple-touch-icon")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESSED)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("missionctrl")
logging.getLogger("uvicorn.access").addFilter(_PollFilter())

# ---------------------------------------------------------------------------
# Singleton environment (lazy-loaded for faster startup)
# ---------------------------------------------------------------------------
_env: Optional[MissionCtrlEnvironment] = None
_completed_results: List[Dict[str, Any]] = []  # Accumulated episode results across tiers
_MAX_LOG_ENTRIES = int(os.getenv("LOG_BUFFER_SIZE", "250"))
_request_logs: deque[RequestLogEntry] = deque(maxlen=_MAX_LOG_ENTRIES)
_build_metadata = BuildMetadata(
    container_id=os.getenv("HOSTNAME", "unknown"),
    build_id=os.getenv("SPACE_ID", os.getenv("BUILD_ID", "unknown")),
    git_sha=os.getenv("GIT_SHA", os.getenv("HF_SPACE_COMMIT_SHA", "unknown")),
    started_at=datetime.now(timezone.utc),
)
_started_at_monotonic = time.monotonic()
_APP_PORT = int(os.getenv("PORT", "7860"))
_APP_HOST = os.getenv("HOST", "0.0.0.0")


def _get_env() -> MissionCtrlEnvironment:
    """Lazy-load the environment on first access."""
    global _env
    if _env is None:
        _env = MissionCtrlEnvironment()
    return _env

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


class StepRequestBody(BaseModel):
    action: str


class ResultRequest(BaseModel):
    tier: str
    score: float
    steps: int
    history: List[Dict[str, Any]] = []
    score_breakdown: Dict[str, Any] = {}
    hallucination_stats: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🛡️  MissionCtrl  v1.0.0                                    ║
    ║   AI Oversight Fleet Environment                             ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                               ║
    ║                                                              ║
    ║   Endpoints:                                                 ║
    ║     GET  /health     → Health check                          ║
    ║     POST /reset      → Reset environment for a task          ║
    ║     POST /step       → Advance engine by 1 action            ║
    ║     GET  /state      → Current observation (read-only)       ║
    ║     GET  /dashboard  → Live visualization UI                 ║
    ║     GET  /history    → Agent action history (JSON)           ║
    ║                                                              ║
    ║   Tasks: easy, medium, hard, special                         ║
    ║                                                              ║
    ║   ┌──────────────────────────────────────────────────────┐   ║
    ║   │  📊 Live Dashboard: http://localhost:{_APP_PORT}/dashboard   │   ║
    ║   └──────────────────────────────────────────────────────┘   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    log.info("Server started — using persistent singleton environment")
    _request_logs.append(
        RequestLogEntry(
            timestamp=datetime.now(timezone.utc),
            method="SYSTEM",
            path="/startup",
            status_code=200,
            duration_ms=0.0,
            container_id=_build_metadata.container_id,
        )
    )
    yield


app = FastAPI(
    title="MissionCtrl",
    version="1.0.0",
    description="AI Oversight Fleet Environment — trains an LLM overseer to detect agent hallucinations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_logger(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    _request_logs.append(
        RequestLogEntry(
            timestamp=datetime.now(timezone.utc),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            container_id=_build_metadata.container_id,
        )
    )
    return response


def _heartbeat_payload(details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = HeartbeatResponse(
        container_id=_build_metadata.container_id,
        build_id=_build_metadata.build_id,
        git_sha=_build_metadata.git_sha,
        runtime=_build_metadata.runtime,
        host=_APP_HOST,
        port=_APP_PORT,
        uptime_seconds=round(time.monotonic() - _started_at_monotonic, 3),
        timestamp_utc=datetime.now(timezone.utc),
        details=details or {},
    )
    return payload.model_dump(mode="json")


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint — simple success response for Hugging Face Spaces."""
    log.debug("Root endpoint accessed")
    return {
        "status": "ok",
        "name": "missionctrl",
        "endpoints": ["/health", "/reset", "/step", "/state", "/dashboard", "/history", "/web", "/ports"],
        "heartbeat": _heartbeat_payload(),
        "log_summary": {"entries": len(_request_logs), "errors": sum(1 for e in _request_logs if e.status_code >= 400)},
        "uptime_seconds": round(time.monotonic() - _started_at_monotonic, 3),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Simple health check for Hugging Face Spaces - returns instantly."""
    log.debug("Health check accessed")
    return {
        "healthy": True,
        **_heartbeat_payload(),
    }


@app.get("/web")
async def web_info() -> Dict[str, Any]:
    return {
        **_heartbeat_payload({
            "dashboard": "/dashboard",
            "logs": "/logs",
        }),
    }


@app.get("/ports")
async def ports() -> Dict[str, Any]:
    return _heartbeat_payload({
        "role": "port_info",
        "configured_port": _APP_PORT,
        "host_binding": _APP_HOST,
        "known_open_ports": [_APP_PORT],
    })


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------
@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    if req is None:
        req = ResetRequest()
    valid = {"easy", "medium", "hard", "special"}
    if req.task_id not in valid:
        raise HTTPException(status_code=422, detail=f"task_id must be one of {sorted(valid)}")

    # Results are now pushed explicitly via POST /result from inference.py
    result = _get_env().reset(task_id=req.task_id, seed=req.seed)
    log.info("Reset → task=%s seed=%s", req.task_id, req.seed)
    return result


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step")
async def step(req: StepRequestBody) -> Dict[str, Any]:
    result = _get_env().step(req.action)
    log.info(
        "Step %d | action=%s | reward=%+.2f done=%s",
        result["observation"]["time_step"],
        req.action[:60],
        result["reward"],
        result["done"],
    )
    return result


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------
@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return current observation snapshot with runtime metadata."""
    observation = _get_env().engine.get_state()
    return {
        **_heartbeat_payload({"role": "state"}),
        "build": _build_metadata.model_dump(mode="json"),
        **observation,
    }


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------
@app.get("/logs")
async def logs() -> LogsSummaryResponse:
    entries = list(_request_logs)
    status_counter = Counter(str(entry.status_code) for entry in entries)
    path_counter = Counter(entry.path for entry in entries)
    totals = {
        "entries": len(entries),
        "unique_paths": len(path_counter),
        "errors": sum(1 for e in entries if e.status_code >= 400),
    }
    return LogsSummaryResponse(
        build=_build_metadata,
        totals=totals,
        statuses=dict(status_counter),
        paths=dict(path_counter),
        entries=entries[-50:],
    )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@app.get("/history")
async def history() -> List[Dict[str, Any]]:
    return list(_get_env().action_history)


# ---------------------------------------------------------------------------
# GET /results — accumulated episode results across tiers
# ---------------------------------------------------------------------------
@app.get("/results")
async def results() -> List[Dict[str, Any]]:
    """Return all completed episode results (persists across resets)."""
    return _completed_results


# ---------------------------------------------------------------------------
# POST /record — explicitly push a completed episode result to the dashboard
# ---------------------------------------------------------------------------
@app.post("/record")
async def record_result(req: ResultRequest) -> Dict[str, str]:
    """Accept a completed task result pushed by the inference script."""
    _completed_results.append({
        "tier": req.tier,
        "score": req.score,
        "steps": req.steps,
        "history": req.history,
        "score_breakdown": req.score_breakdown,
        "hallucination_stats": req.hallucination_stats,
    })
    log.info("Result pushed → tier=%s score=%.4f steps=%d", req.tier, req.score, req.steps)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "redesigned-dashboard.html"),
        os.path.join(os.path.dirname(__file__), "dashboard.html"),
    ]

    for html_path in candidates:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    raise HTTPException(status_code=500, detail="Dashboard HTML file not found")


@app.get("/dashboard/ping")
async def dashboard_ping() -> Dict[str, Any]:
    return _heartbeat_payload({
        "role": "dashboard_ping",
        "dashboard_path": "/dashboard",
        "ready": True,
    })


# ---------------------------------------------------------------------------
# Favicon & Apple Touch Icon — suppress 404 noise
# ---------------------------------------------------------------------------
_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<text y=".9em" font-size="80">🛡️</text>
</svg>"""


@app.get("/favicon.ico")
async def favicon():
    return Response(
        content=_FAVICON_SVG.encode(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    import base64
    px = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualIQAAAABJRU5ErkJggg==")
    return Response(content=px, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/apple-touch-icon-precomposed.png")
async def apple_touch_icon_precomposed():
    import base64
    px = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualIQAAAABJRU5ErkJggg==")
    return Response(content=px, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
