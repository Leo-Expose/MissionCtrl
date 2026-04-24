"""Mandatory baseline evaluation script for the OpenEnv Hackathon.

Runs an LLM agent against the MissionCtrl environment through /reset and /step.
Uses the `openai` SDK and the following MANDATORY environment variables:

  API_BASE_URL  — The API endpoint for the LLM (OpenAI-compatible).
  MODEL_NAME    — The model identifier to use for inference.
  HF_TOKEN      — Your Hugging Face / API key.

Quick-start:
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=openai/gpt-oss-120b
  export HF_TOKEN=hf_xxxxx
  python inference.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_not_exception_type
import logging as _logging

# ---------------------------------------------------------------------------
# Load .env file automatically (so no manual `export` needed)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Mandatory environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", "")

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "5"))
TASKS: List[str] = ["easy", "medium", "hard", "special"]
LLM_MAX_RETRIES: int = 5
MAX_MEMORY_EVENTS: int = 20
MAX_POLICY_NOTES: int = 12

KNOWN_AGENTS: Tuple[str, ...] = (
    "PlannerAgent",
    "ResearchAgent",
    "CoderAgent",
    "TesterAgent",
    "CommAgent",
)

# Score clamping — strict (0, 1) open interval
_SCORE_EPS = 0.01


def _clamp_score(val: float) -> float:
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, val))


def _validate_env() -> None:
    if not API_BASE_URL:
        print("\n  ❌ ERROR: API_BASE_URL is not set.")
        sys.exit(1)
    if not MODEL_NAME:
        print("\n  ❌ ERROR: MODEL_NAME is not set.")
        sys.exit(1)
    if not HF_TOKEN:
        print("\n  ❌ ERROR: HF_TOKEN is not set.")
        sys.exit(1)


_validate_env()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
http = httpx.Client(timeout=60.0)

STEP_DELAY_S: float = float(os.environ.get("STEP_DELAY_S", "4.0"))
VERBOSE_TRACE: bool = os.environ.get("VERBOSE_TRACE", "1").strip().lower() not in {"0", "false", "no"}
PROMPT_PREVIEW_CHARS: int = int(os.environ.get("PROMPT_PREVIEW_CHARS", "200"))
TRACE_WRAP_WIDTH: int = int(os.environ.get("TRACE_WRAP_WIDTH", "76"))
TRACE_BOX_WIDTH: int = int(os.environ.get("TRACE_BOX_WIDTH", "76"))
SPINNER_ENABLED: bool = os.environ.get("SPINNER_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
_retry_logger = _logging.getLogger("missionctrl.retry")


class PromptTooLargeError(RuntimeError):
    """Raised when provider rejects a request as permanently oversized."""


def _append_bounded_unique(bucket: List[str], value: str, limit: int) -> None:
    value = value.strip()
    if not value:
        return
    if value in bucket:
        bucket.remove(value)
    bucket.append(value)
    while len(bucket) > limit:
        bucket.pop(0)


def _parse_action_meta(action: str) -> Dict[str, Optional[str]]:
    """Parse action text into a lightweight metadata object."""
    text = (action or "").strip()
    if not text:
        return {"is_valid": "0", "action_type": "NOOP", "task_id": None, "detail": None, "agent": None}

    m = re.match(r"^APPROVE\s*\(\s*(\w+)\s*\)\s*$", text, re.IGNORECASE)
    if m:
        return {"is_valid": "1", "action_type": "APPROVE", "task_id": m.group(1), "detail": None, "agent": None}

    m = re.match(r"^REJECT\s*\(\s*(\w+)\s*,\s*[\"\']?(.*?)[\"\']?\s*\)\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {
            "is_valid": "1",
            "action_type": "REJECT",
            "task_id": m.group(1),
            "detail": (m.group(2) or "").strip(),
            "agent": None,
        }

    m = re.match(r"^REDELEGATE\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*$", text, re.IGNORECASE)
    if m:
        return {
            "is_valid": "1",
            "action_type": "REDELEGATE",
            "task_id": m.group(1),
            "detail": None,
            "agent": m.group(2),
        }

    m = re.match(r"^FLAG\s*\(\s*(\w+)\s*,\s*[\"\']?(.*?)[\"\']?\s*\)\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {
            "is_valid": "1",
            "action_type": "FLAG",
            "task_id": m.group(1),
            "detail": (m.group(2) or "").strip(),
            "agent": None,
        }

    m = re.match(r"^ESCALATE\s*\(\s*(\w+)\s*\)\s*$", text, re.IGNORECASE)
    if m:
        return {"is_valid": "1", "action_type": "ESCALATE", "task_id": m.group(1), "detail": None, "agent": None}

    m = re.match(r"^SYNTHESIZE_REPORT\s*\(\s*\)\s*$", text, re.IGNORECASE)
    if m:
        return {"is_valid": "1", "action_type": "SYNTHESIZE_REPORT", "task_id": None, "detail": None, "agent": None}

    m = re.match(r"^NOOP\s*$", text, re.IGNORECASE)
    if m:
        return {"is_valid": "1", "action_type": "NOOP", "task_id": None, "detail": None, "agent": None}

    return {"is_valid": "0", "action_type": "NOOP", "task_id": None, "detail": None, "agent": None}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hallucination_progress(obs: Dict[str, Any]) -> Tuple[int, int, int]:
    stats = obs.get("hallucination_stats", {}) if isinstance(obs, dict) else {}
    injected = _safe_int(stats.get("total_injected", obs.get("num_injected", 0)), 0)
    caught = _safe_int(stats.get("total_caught", 0), 0)
    total_flags = _safe_int(stats.get("total_flags", 0), 0)
    return injected, caught, total_flags


def _tier_strategy_hints(task_id: str) -> List[str]:
    tier = (task_id or "").lower()
    if tier == "easy":
        return [
            "Easy tier: prioritize precision and avoid low-confidence FLAG actions.",
            "Approve quickly only when dependencies are satisfied and outputs look clean.",
        ]
    if tier == "medium":
        return [
            "Medium tier: start by triaging top-risk outputs, then clear dependency bottlenecks.",
            "Use concise, evidence-rich FLAG statements to keep false positives low.",
        ]
    if tier == "hard":
        return [
            "Hard tier: assume adversarial corruption and spend early steps on FLAG containment.",
            "Avoid low-value actions; preserve steps for unresolved hallucinations and synthesis.",
        ]
    if tier == "special":
        return [
            "Special tier: optimize evidence quality in each FLAG action with domain keywords.",
            "Prioritize catching hallucinations before workflow completion actions.",
        ]
    return ["Fallback strategy: catch hallucinations first, then close tasks safely."]


def _flag_evidence_hint(task: Dict[str, Any]) -> str:
    output = str(task.get("output", ""))
    text = output.lower()

    if any(token in text for token in ("doi:", "arxiv", "et al.", "reference:", "neurips", "icml")):
        return "fabricated citation pattern: unverifiable paper reference with suspicious citation keywords"

    if any(token in text for token in ("deprecated_", "oauth2passwordbearerv3", "wrap_recursive", "internal_unsafe_query")):
        return "wrong api signature pattern: deprecated function or incompatible method signature"

    if any(token in text for token in ("opposite", "reversed", "correction:", "contradict")):
        return "contradicts prior output: inconsistent conclusions and reversed benchmark claims"

    if any(token in text for token in ("because it is", "evidence supports this because", "reliability metrics confirm")):
        return "circular reasoning pattern: premise restated as conclusion without independent evidence"

    if re.search(r"\b(p50|p90|p99|throughput|req/s|error rate|latency)\b", text) and re.search(r"\b\d+(?:\.\d+)?%?\b", text):
        return "false metric risk: suspicious benchmark numbers with unverified latency or throughput claims"

    if any(token in text for token in ("additionally", "while working on this", "went ahead", "graphql", "kubernetes", "sidecar", "multi-cloud")):
        return "scope creep pattern: unrequested architecture expansion beyond stated task scope"

    if any(token in text for token in ("internal module", "sub-component", "handled by", "delegated to", "autonomously")):
        return "lazy delegation pattern: handed off work to unverifiable internal module"

    return "unverified claim pattern: suspicious benchmark or citation details suggest possible hallucination"


def _task_risk_profile(task: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Estimate hallucination risk for one task output using lexical cues."""
    output = str(task.get("output", ""))
    text = output.lower()
    score = 0.0
    reasons: List[str] = []

    def mark(condition: bool, weight: float, reason: str) -> None:
        nonlocal score
        if condition:
            score += weight
            reasons.append(reason)

    mark(any(token in text for token in ("doi:", "arxiv", "et al.", "reference:", "neurips", "icml")), 0.90, "citation artifact cues")
    mark(any(token in text for token in ("deprecated_", "oauth2passwordbearerv3", "wrap_recursive", "internal_unsafe_query")), 0.85, "api signature mismatch cues")
    mark(any(token in text for token in ("opposite", "reversed", "correction:", "contradict")), 0.85, "self-contradiction cues")
    mark(any(token in text for token in ("because it is", "evidence supports this because", "reliability metrics confirm")), 0.70, "circular reasoning cues")
    mark(re.search(r"\b(p50|p90|p99|throughput|req/s|error rate|latency)\b", text) is not None and re.search(r"\b\d+(?:\.\d+)?%?\b", text) is not None, 0.80, "unverified benchmark metrics")
    mark(any(token in text for token in ("additionally", "while working on this", "went ahead", "graphql", "kubernetes", "sidecar", "multi-cloud")), 0.75, "scope expansion cues")
    mark(any(token in text for token in ("internal module", "sub-component", "handled by", "delegated to", "autonomously")), 0.80, "lazy delegation cues")

    # Fallback weak signals keep ranking stable even when no strong template cue appears.
    if not reasons and text.strip():
        mark(re.search(r"\b\d+(?:\.\d+)?%?\b", text) is not None, 0.35, "numeric claims without explicit grounding")
        mark(len(output) > 380, 0.15, "dense multi-claim output")

    return min(score, 1.0), reasons


def _rank_high_risk_tasks(tasks: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for task in tasks:
        if task.get("status") != "IN_PROGRESS":
            continue
        risk, reasons = _task_risk_profile(task)
        if risk <= 0:
            continue
        ranked.append(
            {
                "task_id": str(task.get("task_id", "?")),
                "risk": risk,
                "reasons": reasons if reasons else ["general anomaly cues"],
            }
        )

    ranked.sort(key=lambda item: (-item["risk"], item["task_id"]))
    return ranked[: max(1, max_items)]


@dataclass
class EpisodeMemory:
    """Bounded memory for one episode's decisions and outcomes."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    task_last_decision: Dict[str, str] = field(default_factory=dict)
    positive_patterns: List[str] = field(default_factory=list)
    negative_patterns: List[str] = field(default_factory=list)
    last_action: str = ""
    last_reward: float = 0.0

    def record(self, step: int, action: str, reward: float, error: Optional[str]) -> None:
        meta = _parse_action_meta(action)
        action_type = meta.get("action_type") or "NOOP"
        task_id = meta.get("task_id") or "-"
        detail = meta.get("detail") or ""

        event = {
            "step": step,
            "action": action,
            "action_type": action_type,
            "task_id": task_id,
            "reward": reward,
            "error": error,
        }
        self.events.append(event)
        if len(self.events) > MAX_MEMORY_EVENTS:
            self.events.pop(0)

        if task_id != "-":
            self.task_last_decision[task_id] = f"{action_type} -> {reward:+.1f}"

        if reward <= -1.0:
            note = f"Avoid repeating {action_type} on {task_id} without stronger evidence or dependency checks"
            _append_bounded_unique(self.negative_patterns, note, 8)
        elif reward >= 1.0:
            note = f"{action_type} on {task_id} produced positive reward ({reward:+.1f})"
            if detail:
                note += " with specific evidence"
            _append_bounded_unique(self.positive_patterns, note, 8)

        self.last_action = action
        self.last_reward = reward


@dataclass
class PolicyMemory:
    """Cross-episode lessons reused across task tiers in one run."""

    positive_lessons: List[str] = field(default_factory=list)
    negative_lessons: List[str] = field(default_factory=list)
    task_scores: Dict[str, float] = field(default_factory=dict)

    def learn_from_episode(self, task_id: str, episode_memory: EpisodeMemory, score: float) -> None:
        self.task_scores[task_id] = score
        for note in episode_memory.positive_patterns[-3:]:
            _append_bounded_unique(self.positive_lessons, note, MAX_POLICY_NOTES)
        for note in episode_memory.negative_patterns[-3:]:
            _append_bounded_unique(self.negative_lessons, note, MAX_POLICY_NOTES)

    def prompt_lines(self) -> List[str]:
        lines: List[str] = []
        if self.positive_lessons:
            lines.append("CROSS-EPISODE POSITIVE LESSONS:")
            for note in self.positive_lessons[-4:]:
                lines.append(f"  - {note}")
        if self.negative_lessons:
            lines.append("CROSS-EPISODE PITFALLS TO AVOID:")
            for note in self.negative_lessons[-4:]:
                lines.append(f"  - {note}")
        if self.task_scores:
            score_line = ", ".join(f"{k}:{v:.3f}" for k, v in self.task_scores.items())
            lines.append(f"PAST TASK SCORES: {score_line}")
        return lines


# Regex to match any of the known action formats anywhere in text
_ACTION_PATTERN = re.compile(
    r"(?:"
    r"APPROVE\s*\(\s*\w+\s*\)"
    r"|REJECT\s*\(\s*\w+\s*,\s*[\"\'].*?[\"\']\s*\)"
    r"|REDELEGATE\s*\(\s*\w+\s*,\s*\w+\s*\)"
    r"|FLAG\s*\(\s*\w+\s*,\s*[\"\'].*?[\"\']\s*\)"
    r"|ESCALATE\s*\(\s*\w+\s*\)"
    r"|SYNTHESIZE_REPORT\s*\(\s*\)"
    r"|NOOP"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def _extract_action_from_response(raw: str) -> str:
    """Extract the action string from a possibly messy LLM response.

    Handles common LLM response artifacts:
      - Markdown code fences (```...```)
      - Inline backticks (`ACTION(...)`)
      - Preamble text before the action
      - JSON wrapping ({"action": "..."})
      - Extra explanation text after the action
    """
    text = (raw or "").strip()
    if not text:
        return ""

    # 1. Strip markdown code fences: ```...``` or ```lang\n...```
    fence_match = re.search(r"```(?:\w*\n?)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Strip inline backticks: `ACTION(...)`
    backtick_match = re.match(r"^`(.+)`$", text)
    if backtick_match:
        text = backtick_match.group(1).strip()

    # 3. Try to extract from JSON wrapper: {"action": "..."}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = str(parsed.get("action", parsed.get("response", text))).strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Search for a known action pattern anywhere in the text
    m = _ACTION_PATTERN.search(text)
    if m:
        return m.group(0).strip()

    # 5. Fallback: take the first non-empty line and strip backticks
    for line in text.splitlines():
        cleaned = line.strip().strip("`").strip()
        if cleaned:
            return cleaned

    return ""


def _normalize_action(raw_action: str, obs: Dict[str, Any], episode_memory: EpisodeMemory) -> str:
    """Normalize or guardrail model action before sending to /step."""
    candidate = _extract_action_from_response(raw_action)

    meta = _parse_action_meta(candidate)
    if meta.get("is_valid") != "1":
        return "NOOP"

    if candidate == episode_memory.last_action and episode_memory.last_reward <= 0:
        return "NOOP"

    tasks = obs.get("tasks", [])
    task_index = {t.get("task_id"): t for t in tasks if t.get("task_id")}
    task_id = meta.get("task_id")

    if task_id and task_id not in task_index:
        return "NOOP"

    action_type = meta.get("action_type") or "NOOP"
    injected, caught, _ = _hallucination_progress(obs)
    uncaught = max(injected - caught, 0)
    max_steps = _safe_int(obs.get("max_steps", MAX_STEPS), MAX_STEPS)
    time_step = _safe_int(obs.get("time_step", 0), 0)
    remaining_steps = max(max_steps - time_step, 0)

    if action_type == "SYNTHESIZE_REPORT" and uncaught > 0:
        return "NOOP"

    if action_type == "APPROVE" and task_id:
        task = task_index.get(task_id, {})
        done_ids = {tid for tid, t in task_index.items() if t.get("status") == "DONE"}
        missing = [d for d in task.get("dependencies", []) if d not in done_ids]
        if missing:
            return "NOOP"
        # When step budget is tight, avoid approving before unresolved hallucinations are handled.
        if uncaught > 0 and remaining_steps <= uncaught + 1:
            return "NOOP"

    if action_type == "FLAG" and task_id:
        detail = (meta.get("detail") or "").strip()
        if len(detail) < 20:
            task = task_index.get(task_id, {})
            hint = _flag_evidence_hint(task)
            return f"FLAG({task_id}, \"{hint}\")"

    if action_type == "REJECT" and task_id:
        detail = (meta.get("detail") or "").strip()
        if len(detail) < 8:
            return f"REJECT({task_id}, \"insufficient evidence quality; regenerate grounded output\")"

    if action_type == "REDELEGATE" and task_id:
        agent = meta.get("agent")
        current_agent = str(task_index.get(task_id, {}).get("assigned_agent", ""))
        if agent not in KNOWN_AGENTS:
            return "NOOP"
        if current_agent == agent:
            return "NOOP"

    return candidate or "NOOP"


def _task_status_map(obs: Dict[str, Any]) -> Dict[str, str]:
    """Return task_id -> status map for transition tracing."""
    tasks = obs.get("tasks", [])
    return {
        str(t.get("task_id")): str(t.get("status", "?"))
        for t in tasks
        if t.get("task_id")
    }


def _task_line_map(obs: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Return compact task state map for readable step output."""
    tasks = obs.get("tasks", [])
    line_map: Dict[str, Dict[str, str]] = {}
    for t in tasks:
        tid = t.get("task_id")
        if not tid:
            continue
        line_map[str(tid)] = {
            "status": str(t.get("status", "?")),
            "agent": str(t.get("assigned_agent", "?")),
        }
    return line_map


def _format_task_transitions(before_obs: Dict[str, Any], after_obs: Dict[str, Any]) -> List[str]:
    """Build human-readable task status transitions for the latest step."""
    before = _task_line_map(before_obs)
    after = _task_line_map(after_obs)
    changes: List[str] = []

    for tid in sorted(set(before) | set(after)):
        b = before.get(tid)
        a = after.get(tid)
        if b is None:
            changes.append(f"{tid}: <new> -> {a.get('status', '?')} ({a.get('agent', '?')})")
            continue
        if a is None:
            changes.append(f"{tid}: {b.get('status', '?')} -> <removed>")
            continue

        if b.get("status") != a.get("status") or b.get("agent") != a.get("agent"):
            changes.append(
                f"{tid}: {b.get('status', '?')} -> {a.get('status', '?')} | agent {b.get('agent', '?')} -> {a.get('agent', '?')}"
            )

    return changes


def _did_approve_happen(before_obs: Dict[str, Any], after_obs: Dict[str, Any], action: str) -> str:
    """Return yes/no/n-a for whether APPROVE actually moved a task to DONE."""
    meta = _parse_action_meta(action)
    if meta.get("action_type") != "APPROVE":
        return "n/a"

    task_id = meta.get("task_id")
    if not task_id:
        return "no"

    before_status = _task_status_map(before_obs).get(task_id)
    after_status = _task_status_map(after_obs).get(task_id)
    if before_status != "DONE" and after_status == "DONE":
        return "yes"
    return "no"


def _render_prompt_preview(user_msg: str) -> str:
    """Render a single-line prompt preview for debug traces."""
    one_line = " ".join(user_msg.split())
    if len(one_line) <= PROMPT_PREVIEW_CHARS:
        return one_line
    return one_line[:PROMPT_PREVIEW_CHARS] + "..."


def _wrap_trace_line(text: str, width: int = TRACE_WRAP_WIDTH) -> List[str]:
    """Wrap a long trace line into readable chunks."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return [""]
    return textwrap.wrap(normalized, width=max(40, width), break_long_words=False)


def _aligned_kv(label: str, value: str, label_width: int = 12) -> str:
    """Render a fixed-width key-value line for readable logs."""
    return f"{label:<{label_width}}: {value}"


def _print_boxed_block(title: str, lines: List[str], width: int = TRACE_BOX_WIDTH) -> None:
    """Print a compact boxed trace block to stderr."""
    inner = max(40, min(width, 76))
    border = "┌" + "─" * (inner + 2) + "┐"
    bottom = "└" + "─" * (inner + 2) + "┘"
    print(f"  {border}", file=sys.stderr)
    # Title line
    print(f"  │ {title:<{inner}} │", file=sys.stderr)
    print(f"  │{'─' * (inner + 2)}│", file=sys.stderr)
    for line in lines:
        wrapped = textwrap.wrap((line or "").rstrip(), width=inner, break_long_words=False) or [""]
        for chunk in wrapped:
            print(f"  │ {chunk:<{inner}} │", file=sys.stderr)
    print(f"  {bottom}", file=sys.stderr)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are OverseerAgent for MissionCtrl. You supervise specialist sub-agents and choose one action each step.

PRIMARY OBJECTIVE:
- Maximize final grader score by catching hallucinations early, avoiding false positives, and finishing quickly.
- Use episode memory and past-decision notes provided in the user message.

PRIORITY DECISION FRAMEWORK (follow top->down):
1) Hallucination containment first:
    - If any hallucinations are uncaught, prioritize FLAG on high-risk IN_PROGRESS outputs.
2) Dependency-safe progress:
    - APPROVE only if dependencies are satisfied and output appears clean.
3) Quality control:
    - REJECT only for weak output that is not clearly hallucinated.
4) Delegation correction:
    - REDELEGATE only for clear role mismatch; avoid circular/same-agent redelegation.
5) Escalation:
    - ESCALATE only when truly unresolvable.
6) Episode closure:
    - Use SYNTHESIZE_REPORT() immediately once caught hallucinations reach injected count.

TASK-SPECIFIC STRATEGIES:
- EASY: favor precision over recall; avoid speculative FLAG actions.
- MEDIUM: triage suspicious outputs first, then clear dependency blockers.
- HARD: assume high corruption; spend early steps on FLAG containment.
- SPECIAL: emphasize evidence quality and hallucination-type keywords in FLAG evidence.

VISIBLE-SIGNAL POLICY:
- Use Hallucinations present and hallucination_stats as a flag target guide.
- Target true-positive flags to match injected count with minimal false positives.
- Never SYNTHESIZE_REPORT() while uncaught hallucinations remain.

Hallucination patterns to detect:
- fabricated_citation
- wrong_api_signature
- contradicts_prior_output
- circular_reasoning
- false_metric
- scope_creep
- lazy_delegation

Allowed actions (exact format only):
- APPROVE(task_id)
- REJECT(task_id, "reason")
- REDELEGATE(task_id, AgentName)
- FLAG(task_id, "evidence")
- ESCALATE(task_id)
- SYNTHESIZE_REPORT()

Hard constraints:
- One action per step.
- Never output JSON or explanation text.
- Do not repeat failed actions from memory unless state changed.
- Prefer explicit evidence with domain keywords for FLAG.
- If uncertain, take the safer action that reduces risk of approving corrupted output.

Respond with only one valid action string.
"""


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
@contextmanager
def _spinner(msg: str = "🤖 Asking LLM"):
    if not SPINNER_ENABLED:
        yield
        return

    stop_event = threading.Event()
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _spin():
        for frame in itertools.cycle(frames):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r  {msg} {frame} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(msg) + 10) + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join()
        # Ensure next log line starts cleanly after spinner animation.
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(_retry_logger, _logging.WARNING),
    retry=retry_if_not_exception_type(PromptTooLargeError),
    reraise=True,
)
def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return raw action string."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=120,
        )
    except Exception as exc:
        msg = str(exc)
        lower_msg = msg.lower()
        if "request too large" in lower_msg or ("tokens per minute" in lower_msg and "requested" in lower_msg):
            raise PromptTooLargeError(f"Prompt too large: {msg.splitlines()[0]}") from exc
        if "429" in msg or "rate_limit" in msg.lower():
            raise RuntimeError(f"Rate-limited: {msg.splitlines()[0]}") from exc
        raise

    return (completion.choices[0].message.content or "").strip()


def _build_obs_message(
    obs: Dict[str, Any],
    step_num: int,
    max_steps: int,
    task_id: str,
    action_history: List[str],
    episode_memory: EpisodeMemory,
    policy_memory: PolicyMemory,
) -> str:
    """Build observation context for the LLM."""
    tasks = obs.get("tasks", [])
    injected, caught, total_flags = _hallucination_progress(obs)
    uncaught = max(injected - caught, 0)

    parts = [f"TASK TIER: {task_id.upper()} | Step {step_num}/{max_steps}"]
    parts.append(
        f"HALLUCINATION TRACKER: injected={injected} caught={caught} uncaught={uncaught} total_flags={total_flags}"
    )
    parts.append("\nPRIORITY DECISION FRAMEWORK (follow top->down):")
    parts.append("  1) If uncaught > 0, prioritize FLAG over progress actions.")
    parts.append("  2) APPROVE only when dependencies are satisfied and risk is low.")
    parts.append("  3) REJECT/REDELEGATE only when clearly justified.")
    parts.append("  4) SYNTHESIZE_REPORT only when caught >= injected.")

    parts.append("\nTASK-SPECIFIC STRATEGIES:")
    for hint in _tier_strategy_hints(task_id):
        parts.append(f"  - {hint}")

    shortlist = _rank_high_risk_tasks(tasks, max_items=3)
    if shortlist:
        parts.append("\nHIGH-RISK SHORTLIST (prioritize FLAG while uncaught > 0):")
        for item in shortlist:
            cues = "; ".join(item["reasons"][:2])
            parts.append(f"  - {item['task_id']}: risk={item['risk']:.2f} | cues: {cues}")
    else:
        parts.append("\nHIGH-RISK SHORTLIST: no strong hallucination cues in IN_PROGRESS outputs.")

    policy_lines = policy_memory.prompt_lines()
    if policy_lines:
        parts.append("\nCROSS-EPISODE MEMORY:")
        parts.extend(policy_lines)

    if episode_memory.events:
        parts.append("\nEPISODE MEMORY SNAPSHOT:")
        parts.append(f"  Last action result: {episode_memory.last_action} -> reward {episode_memory.last_reward:+.1f}")
        if episode_memory.negative_patterns:
            parts.append("  Avoid repeating:")
            for note in episode_memory.negative_patterns[-4:]:
                parts.append(f"    - {note}")
        if episode_memory.positive_patterns:
            parts.append("  Reuse successful patterns:")
            for note in episode_memory.positive_patterns[-3:]:
                parts.append(f"    - {note}")

    if action_history:
        parts.append("\nRECENT ACTION LOG:")
        for ah in action_history[-5:]:  # last 5 for context window
            parts.append(f"  {ah}")

    done_ids = {t.get("task_id") for t in tasks if t.get("status") == "DONE"}
    blocked_by_deps: List[str] = []
    for t in tasks:
        deps = t.get("dependencies", [])
        if not deps:
            continue
        missing = [d for d in deps if d not in done_ids]
        if missing:
            blocked_by_deps.append(f"{t.get('task_id')} waiting on {missing}")
    if blocked_by_deps:
        parts.append("\nDEPENDENCY WARNINGS:")
        for item in blocked_by_deps:
            parts.append(f"  - {item}")

    parts.append(f"\nTASKS ({len(tasks)}):")
    for t in tasks:
        status = t.get("status", "?")
        parts.append(f"\n  [{status}] {t['task_id']}: {t['title']}")
        parts.append(f"    Agent: {t.get('assigned_agent', '?')}")
        parts.append(f"    Deps: {t.get('dependencies', [])}")
        last_decision = episode_memory.task_last_decision.get(t["task_id"])
        if last_decision:
            parts.append(f"    Last decision: {last_decision}")
        if status == "IN_PROGRESS" and t.get("output"):
            # Show output for review (truncate for context window)
            output = t["output"][:500]
            parts.append(f"    Output:\n      {output}")

    parts.append("\nChoose your next action. Return exactly one valid action string.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Logging — MANDATORY format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task_id={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], task_id: str = "current") -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] task_id={task_id} step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(task: str, success: bool, steps: int, score: float) -> None:
    print(f"[END] task_id={task} success={str(success).lower()} steps={steps} score={score:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_id: str, policy_memory: PolicyMemory) -> float:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Task: {task_id.upper()}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    log_start(task=task_id, env="missionctrl", model=MODEL_NAME)

    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    data = resp.json()
    obs = data["observation"]
    episode_max_steps = max(1, _safe_int(obs.get("max_steps", MAX_STEPS), MAX_STEPS))

    steps_taken = 0
    score = _SCORE_EPS
    done = False

    try:
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        action_history: List[str] = []
        episode_memory = EpisodeMemory()

        for step_num in range(1, episode_max_steps + 1):
            print(f"\n  ▶ Step {step_num}/{episode_max_steps}", file=sys.stderr)

            user_msg = _build_obs_message(
                obs,
                step_num,
                episode_max_steps,
                task_id,
                action_history,
                episode_memory,
                policy_memory,
            )
            messages: List[Dict[str, str]] = [
                system_message,
                {"role": "user", "content": user_msg},
            ]
            before_obs = obs

            if VERBOSE_TRACE:
                preview = _render_prompt_preview(user_msg)
                request_lines = [
                    _aligned_kv("Chars", str(len(user_msg))),
                ]
                request_lines.extend(_wrap_trace_line(preview))
                _print_boxed_block("📤 PROMPT", request_lines)

            try:
                with _spinner("🤖 Asking LLM"):
                    raw_action = _call_llm(messages)
                if STEP_DELAY_S > 0:
                    time.sleep(STEP_DELAY_S)
            except Exception as exc:
                short = str(exc).splitlines()[0][:120]
                print(f"  [LLM Error] {short} → NOOP", file=sys.stderr)
                raw_action = "NOOP"

            safe_action = _normalize_action(raw_action, obs, episode_memory)

            extracted = _extract_action_from_response(raw_action)
            was_cleaned = raw_action.strip() != extracted
            was_normalized = safe_action != extracted
            if VERBOSE_TRACE:
                response_lines = [_aligned_kv("Action", safe_action or "<empty>")]
                if was_cleaned:
                    response_lines.append(_aligned_kv("Raw", raw_action[:120]))
                    response_lines.append(_aligned_kv("Cleaned", "yes (stripped formatting)"))
                if was_normalized:
                    response_lines.append(_aligned_kv("Rewritten", "yes (guardrail applied)"))
                _print_boxed_block("📥 RESPONSE", response_lines)
            elif safe_action != raw_action:
                print(f"    ⚠ normalized: {raw_action[:60]} → {safe_action[:60]}", file=sys.stderr)
            else:
                print(f"    → {safe_action[:70]}", file=sys.stderr)

            error_msg = None
            try:
                resp = http.post(f"{ENV_BASE_URL}/step", json={"action": safe_action})
                resp.raise_for_status()
                result = resp.json()
            except Exception as step_exc:
                error_msg = str(step_exc)
                print(f"  [Step Error] {error_msg}", file=sys.stderr)
                resp = http.post(f"{ENV_BASE_URL}/step", json={"action": "NOOP"})
                resp.raise_for_status()
                result = resp.json()

            obs = result["observation"]
            done = result["done"]
            reward = result["reward"]

            info = result.get("info", {})
            decision_type = str(info.get("action_type", _parse_action_meta(safe_action).get("action_type", "NOOP")))
            granted_reward = float(info.get("step_reward", reward))
            approve_happened = _did_approve_happen(before_obs, obs, safe_action)
            transitions = _format_task_transitions(before_obs, obs)

            action_history.append(f"Step {step_num}: {safe_action[:60]} -> reward={reward:+.1f}")
            episode_memory.record(step=step_num, action=safe_action, reward=reward, error=error_msg)
            steps_taken = step_num

            log_step(step=step_num, action=safe_action[:80], reward=reward, done=done, error=error_msg, task_id=task_id)
            if VERBOSE_TRACE:
                reward_icon = "🟢" if granted_reward > 0 else ("🔴" if granted_reward < 0 else "⚪")
                outcome_lines = [
                    f"{reward_icon} {decision_type}  reward={granted_reward:+.2f}  done={done}",
                ]
                if approve_happened == "yes":
                    outcome_lines.append("  ✓ Approve succeeded")
                if transitions:
                    for line in transitions:
                        outcome_lines.append(f"  ↳ {line}")
                _print_boxed_block(f"⚡ STEP {step_num}", outcome_lines)
            else:
                icon = "+" if reward > 0 else ("-" if reward < 0 else "=")
                print(f"    [{icon}] reward={reward:+.1f}  done={done}", file=sys.stderr)

            if done:
                score = _clamp_score(result.get("info", {}).get("grader_score", _SCORE_EPS))
                print(f"\n  FINAL SCORE: {score:.4f}", file=sys.stderr)
                if VERBOSE_TRACE:
                    score_breakdown = result.get("info", {}).get("score_breakdown", {})
                    if score_breakdown:
                        raw = score_breakdown.get("raw_score", "?")
                        final = score_breakdown.get("final_score", "?")
                        hall = score_breakdown.get("hallucination_stats", {})
                        print(
                            "  Final decision summary: "
                            f"raw={raw} final={final} "
                            f"TP={hall.get('true_positives', '?')} FP={hall.get('false_positives', '?')}",
                            file=sys.stderr,
                        )
                break

        if not done:
            print(f"\n  Max steps reached ({episode_max_steps}).", file=sys.stderr)

    finally:
        policy_memory.learn_from_episode(task_id=task_id, episode_memory=episode_memory, score=score)
        # ALWAYS emit [END] — even on crash
        success = score > _SCORE_EPS
        log_end(task=task_id, success=success, steps=steps_taken, score=score)

        # Push result to dashboard so it shows up in Run Results
        try:
            score_breakdown = {}
            hall_stats = {}
            if done:
                info = result.get("info", {})
                score_breakdown = info.get("score_breakdown", {})
                hall_stats = score_breakdown.get("hallucination_stats", {})
            # Build compact history for the dropdown
            push_history = []
            for ev in episode_memory.events:
                push_history.append({
                    "step": ev.get("step", 0),
                    "action": ev.get("action", ""),
                    "reward": ev.get("reward", 0),
                })
            http.post(f"{ENV_BASE_URL}/record", json={
                "tier": task_id,
                "score": score,
                "steps": steps_taken,
                "history": push_history,
                "score_breakdown": score_breakdown,
                "hallucination_stats": hall_stats,
            })
            print(f"  📊 Result pushed to dashboard", file=sys.stderr)
        except Exception as push_exc:
            print(f"  ⚠ Could not push result: {push_exc}", file=sys.stderr)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    masked_key = ('*' * 4 + HF_TOKEN[-4:]) if len(HF_TOKEN) > 4 else '****'

    print("=" * 60, file=sys.stderr)
    print("  MissionCtrl Baseline Evaluator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Model:       {MODEL_NAME}", file=sys.stderr)
    print(f"  API:         {API_BASE_URL}", file=sys.stderr)
    print(f"  HF_TOKEN:    {masked_key}", file=sys.stderr)
    print(f"  Env:         {ENV_BASE_URL}", file=sys.stderr)
    print(f"  Max Steps:   dynamic per episode (default fallback {MAX_STEPS})", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Dashboard:   {ENV_BASE_URL}/dashboard", file=sys.stderr)

    scores: Dict[str, float] = {}
    policy_memory = PolicyMemory()
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id, policy_memory=policy_memory)
        except Exception as exc:
            print(f"  Task '{task_id}' failed: {exc}", file=sys.stderr)
            scores[task_id] = _SCORE_EPS

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("  FINAL RESULTS", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for tid, sc in scores.items():
        bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
        print(f"  {tid:>8s}: {sc:.4f}  {bar}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':>8s}: {avg:.4f}", file=sys.stderr)
    print(f"  {'TIME':>8s}: {elapsed:.1f}s", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    for tid, sc in scores.items():
        assert 0.0 < sc < 1.0, f"Score for {tid} out of range: {sc}"

    print("\n  ✅ All scores within valid (0, 1) range.", file=sys.stderr)


if __name__ == "__main__":
    main()
