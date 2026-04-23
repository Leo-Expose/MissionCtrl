"""FastAPI application for MissionCtrl Environment.

Uses a persistent singleton MissionCtrlEnvironment so that state is preserved
across /reset and /step calls.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
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
# Singleton environment
# ---------------------------------------------------------------------------
_env = MissionCtrlEnvironment()
_completed_results: List[Dict[str, Any]] = []  # Accumulated episode results across tiers

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
    banner = r"""
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
    ║   │  📊 Live Dashboard: http://localhost:8000/dashboard   │   ║
    ║   └──────────────────────────────────────────────────────┘   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    log.info("Server started — using persistent singleton environment")
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


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint — heartbeat for OpenEnv platform probes."""
    return {
        "status": "ok",
        "name": "missionctrl",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/dashboard", "/history"],
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"healthy": True, "env": "missionctrl"}


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
    result = _env.reset(task_id=req.task_id, seed=req.seed)
    log.info("Reset → task=%s seed=%s", req.task_id, req.seed)
    return result


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step")
async def step(req: StepRequestBody) -> Dict[str, Any]:
    result = _env.step(req.action)
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
    """Return current observation snapshot for dashboard."""
    return _env.engine.get_state()


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@app.get("/history")
async def history() -> List[Dict[str, Any]]:
    return list(_env.action_history)


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
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


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
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
