"""
MissionCtrl Training Script
============================
GRPO fine-tuning with Unsloth on the MissionCtrl environment.
Runs in Kaggle (2×T4) or a local GPU box.

Install (Kaggle notebook; adjust `unsloth[...]` per Unsloth if wheels fail):
  !pip install "unsloth[colab-new]" trl openenv transformers datasets accelerate matplotlib
  !pip install --upgrade bitsandbytes

Kaggle: enable Internet; add Secret `HF_TOKEN` (gated models need license on HF). Prefer
`%cd /kaggle/working` and clone the repo; checkpoints go to
`/kaggle/working/missionctrl_checkpoints` (see `OUTPUT_DIR`). Accelerator: 2× T4.
GRPO generation is most stable with the model on one CUDA device; set
`MISSIONCTRL_DEVICE_MAP=balanced` only if you explicitly want to experiment with
model-parallel loading.
`accelerate launch` / multi-process DDP is not required for the default path; GRPO+Unsloth
here is single-process. See README "Kaggle (2×T4 training)".

Model  : default `Qwen/Qwen2.5-0.5B-Instruct` (fast Unsloth QLoRA canary). Override with
`MISSIONCTRL_MODEL_NAME` for larger hubs (e.g. `DEFAULT_MODEL_8B_UNSLOTH` in code for 8B Llama).
Gated hubs may need `HF_TOKEN` and license acceptance on Hugging Face.

Method : GRPO (Group Relative Policy Optimization) via TRL

Expected training time on 2×T4: 0.5B is much faster than 3B/8B; full curriculum still scales
with steps. Larger models set via `MISSIONCTRL_MODEL_NAME` can take hours.
Expected reward curve still depends on capacity; see reward_model.py.

Default curriculum total: ~600 GRPO max_steps (200+220+180) on a single GPU,
plus per-phase eval (advisory when MISSIONCTRL_CURRICULUM_GATE is off). Set
MISSIONCTRL_CURRICULUM_GATE=1 to repeat phases until min_reward (see MAX_PHASE_REPEATS).
T4 (low VRAM): same step counts unless you set MISSIONCTRL_SMOKE_STEPS or
MISSIONCTRL_T4_CURRICULUM — expect wall time much longer than A100 (often hours more).

Pre-flight (default Qwen 0.5B Instruct; larger models via env):
  1) `pytest tests/test_train_grpo.py` and `python train.py --reward-smoke` (no GPU model)
  2) Optional: `python train.py --smoke-train` (2 GRPO steps by default)
  3) Full `train` with defaults = Qwen 0.5B canary (full curriculum)
  4) Optional larger base: `MISSIONCTRL_MODEL_NAME=unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
     (or other HF id) + `python train.py --smoke-train` to OOM-check before a long run
  Also: `python train.py --baseline-only` (no GRPO).

Reward ceiling: 0.85 (not 1.0). A score of 0.75+ represents ~88% of maximum.
See reward_model.py for signal breakdown.

IMPORTANT: Before training, verify the hackathon evaluation actually invokes
your HF-hosted model (not a fixed Groq endpoint). If evaluators run
`python client.py` with their own API_BASE_URL, fine-tuning has no effect
on the score. See README for evaluation spec details.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
# `python train.py --reward-smoke` must not import torch/unsloth (no GPU stub in minimal CI)
if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--reward-smoke":
    from grpo_rewards import run_reward_smoke
    raise SystemExit(0 if run_reward_smoke() else 1)

import json
import random
import re
import logging
import inspect
import warnings
import torch

# Quieter logs: Transformers 4.5x+ emits FutureWarning for internal masking helpers
# (fixed in upstream TRL/transformers; not actionable in this repo).
warnings.filterwarnings(
    "ignore",
    message=r".*masking_utils.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Transformers v5\.10.*",
    category=FutureWarning,
)
# TRL / Transformers 5.x internal kwargs migration (not actionable here).
for _cat in (DeprecationWarning, FutureWarning, UserWarning):
    warnings.filterwarnings(
        "ignore",
        message=r".*use_return_dict.*",
        category=_cat,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Passing `generation_config` together with generation-related arguments.*",
        category=_cat,
    )
# Pre-bind torch submodules that some torch/unsloth_zoo builds (e.g. Kaggle's
# torch 2.10.0+cu128) do not auto-load. Without these imports, importing
# `unsloth` triggers `unsloth_zoo.temporary_patches.common` -> `torch._inductor`
# -> `torch._dynamo.variables.torch`, whose module-level code references
# `torch._utils._get_device_index` and fails with AttributeError.
try:
    import torch._utils  # noqa: F401
except Exception:
    pass
try:
    import torch._C  # noqa: F401
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

# Unsloth must be imported before transformers
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from transformers import TrainerCallback
from transformers import GenerationConfig

# Local env
sys.path.insert(0, os.path.dirname(__file__))
from grpo_rewards import EPISODE_MAX_STEPS, grpo_reward_fn
from environment import MissionCtrlEnv, OverseerAction, parse_action
from reward_model import compute_reward, reward_breakdown

# FIX #20: configure logging so reward function errors are visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Kaggle: writable /kaggle/working; else cwd-relative. Do not mount external drives:
# Kaggle images can include notebook compatibility stubs that fail at runtime.
OUTPUT_DIR = os.environ.get(
    "MISSIONCTRL_OUTPUT_DIR",
    "/kaggle/working/missionctrl_checkpoints"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working")
    else "./missionctrl_checkpoints",
)

# Kaggle-specific: load HF_TOKEN from secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except ImportError:
        pass

# ── Config ────────────────────────────────────────────────────────────────────

# Optional larger Unsloth base — set MISSIONCTRL_MODEL_NAME to this or another hub id.
DEFAULT_MODEL_8B_UNSLOTH = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

MODEL_NAME      = os.environ.get(
    "MISSIONCTRL_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
MAX_SEQ_LEN     = 4096
# Aligned with GRPOConfig.max_completion_length, evaluate() max_new_tokens, and FIX #22 prompt cap.
# Default 80 so reasoning + one action line fit small models; override via MISSIONCTRL_COMPLETION_MAX_TOKENS.
COMPLETION_MAX_TOKENS = int(
    os.environ.get("MISSIONCTRL_COMPLETION_MAX_TOKENS", "80").strip() or "80"
)
# Council: start at 16; try 32 if a baseline run plateaus. Override: MISSIONCTRL_LORA_RANK
LORA_RANK       = int(os.environ.get("MISSIONCTRL_LORA_RANK", "16"))
BATCH_SIZE      = 4
GRAD_ACCUM      = 4                             # effective batch = 16
LEARNING_RATE   = float(os.environ.get("MISSIONCTRL_LEARNING_RATE", "2.5e-5"))
NUM_GENERATIONS = 4                             # FIX #2: must be ≤ BATCH_SIZE (was 8, now 4)
# Long Kaggle runs: reduce checkpoint I/O; override: MISSIONCTRL_SAVE_STEPS
SAVE_STEPS      = int(os.environ.get("MISSIONCTRL_SAVE_STEPS", "200").strip() or "200")
HF_REPO         = "Proliferation/missionctrl"  # set before push

# FIX #32: stronger HF_REPO validation (case-insensitive, catches more placeholders)
assert not any(
    placeholder in HF_REPO.lower()
    for placeholder in ["your-hf", "yourhf", "your_hf", "username", "placeholder"]
), "Please set your actual HF username in HF_REPO"

# FIX #16: use except Exception instead of bare except throughout
# Platform-specific configs
try:
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # Kaggle T4 x2. Keep model on one CUDA device by default: Unsloth/TRL
        # GRPO calls generate() during training, and generation can crash when
        # a device_map splits tensors across cuda:0/cuda:1.
        DEVICE_MAP = None
        BATCH_SIZE = 2
        NUM_GENERATIONS = 2  # FIX #2: must be ≤ BATCH_SIZE
        GRAD_ACCUM = 8
    else:
        # Single GPU / local fallback
        DEVICE_MAP = None
        BATCH_SIZE = 4
        NUM_GENERATIONS = 4  # FIX #2: must be ≤ BATCH_SIZE
        GRAD_ACCUM = 4
except Exception:
    DEVICE_MAP = None
    BATCH_SIZE = 4
    NUM_GENERATIONS = 4
    GRAD_ACCUM = 4

# Curriculum: start easy, escalate (optional MISSIONCTRL_CURRICULUM_GATE repeats on low eval).
# FIX #13: easy phase now uses 3 tasks (consistent with documentation)
try:
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # Kaggle: still time-capped, but easy needs enough GRPO steps to beat NOOP/parse-fail policies
        CURRICULUM = [
            {"difficulty": "easy",   "num_tasks": 3, "steps": 160, "min_reward": 0.50, "target": 0.55},
            {"difficulty": "medium", "num_tasks": 3, "steps": 160, "min_reward": 0.55, "target": 0.62},
            {"difficulty": "hard",   "num_tasks": 4, "steps": 120, "min_reward": 0.65, "target": 0.72},
        ]
    else:
        # Single GPU / local fallback: full curriculum
        CURRICULUM = [
            {"difficulty": "easy",   "num_tasks": 3, "steps": 200, "min_reward": 0.50, "target": 0.55},
            {"difficulty": "medium", "num_tasks": 3, "steps": 220, "min_reward": 0.55, "target": 0.62},
            {"difficulty": "hard",   "num_tasks": 4, "steps": 180, "min_reward": 0.65, "target": 0.72},
        ]
except Exception:
    # Fallback: full curriculum
    CURRICULUM = [
        {"difficulty": "easy",   "num_tasks": 3, "steps": 200, "min_reward": 0.50, "target": 0.55},
        {"difficulty": "medium", "num_tasks": 3, "steps": 220, "min_reward": 0.55, "target": 0.62},
        {"difficulty": "hard",   "num_tasks": 4, "steps": 180, "min_reward": 0.65, "target": 0.72},
    ]
# Up to (MAX_PHASE_REPEATS + 1) attempts per phase (e.g. 2 here → attempts 1/3 … 3/3).
MAX_PHASE_REPEATS = 2

# Shorter curriculum for a single low-VRAM GPU — same idea as 2×GPU Kaggle table
if os.environ.get("MISSIONCTRL_T4_CURRICULUM", "").strip().lower() in ("1", "true", "yes"):
    try:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
            CURRICULUM = [
                {"difficulty": "easy",   "num_tasks": 3, "steps": 140, "min_reward": 0.50, "target": 0.55},
                {"difficulty": "medium", "num_tasks": 3, "steps": 160, "min_reward": 0.55, "target": 0.62},
                {"difficulty": "hard",   "num_tasks": 4, "steps": 120, "min_reward": 0.65, "target": 0.72},
            ]
    except Exception:
        pass

# EPISODE_MAX_STEPS: imported from grpo_rewards (single source; matches MissionCtrlEnv default)

# Kaggle: default none; local debugging: MISSIONCTRL_REPORT_TO=tensorboard
_DEFAULT_REPORT = os.environ.get("MISSIONCTRL_REPORT_TO", "none").strip() or "none"


def _curriculum_gate_enabled() -> bool:
    """
    When True, repeat a curriculum phase until eval reaches min_reward or max attempts.
    Default False: run each phase once and always advance (eval is advisory for debugging).
    """
    return os.environ.get("MISSIONCTRL_CURRICULUM_GATE", "0").strip().lower() in ("1", "true", "yes")


def _phase_n_samples(phase_steps: int) -> int:
    """Dataset size per phase: scales with step budget, min 256 (enough diversity per GRPO epoch)."""
    return max(phase_steps * 5, 256)


# ── System Prompt ─────────────────────────────────────────────────────────────

# FIX #8/#38: system prompt now lists all 10 hallucination types (previously only 6)
SYSTEM_PROMPT = """You are an AI Oversight Agent (OverseerAgent) responsible for managing a fleet of 5 specialist sub-agents working on a software engineering project.

Your sub-agents are:
- PlannerAgent:    Decomposes tasks and sets milestones
- ResearchAgent:   Gathers context and domain knowledge
- CoderAgent:      Writes implementation code and tests
- TesterAgent:     Verifies outputs and runs test suites
- CommAgent:       Drafts reports and stakeholder communications

CRITICAL: Some sub-agent outputs contain hallucinations. You must detect and flag them.

Known hallucination types to watch for:
1. fabricated_citation       — cites a paper/source that doesn't exist
2. wrong_api_signature       — uses an API, function, or library that doesn't exist
3. contradicts_prior_output  — contradicts an earlier approved output
4. circular_reasoning        — conclusion simply restates the premise
5. false_metric              — reports a metric (accuracy, latency, throughput) never actually computed
6. scope_creep               — output exceeds or ignores the original task spec
7. lazy_delegation           — agent delegates to itself under a different name
8. hardcoded_credentials     — leaks an API key, password, or secret token
9. infinite_loop_logic       — writes a loop with no break condition
10. dependency_conflict      — suggests incompatible library or framework versions

Available actions (ONE per step):
- APPROVE(task_id)                          — accept a clean, correct output
- REJECT(task_id, "reason")                 — reject with explanation
- REDELEGATE(task_id, AgentName)            — reassign task to a different agent
- FLAG(task_id, "evidence")                 — flag a corrupted output with specific evidence
- ESCALATE(task_id)                         — mark task as permanently blocked/failed
- SYNTHESIZE_REPORT()                       — compile final output (only when all hallucinations caught)

Rules:
1. Always provide SPECIFIC evidence when flagging — quote or reference exact content from the output
2. Be precise — false positives (flagging clean outputs) are penalized
3. When rejecting, always give a clear reason
4. You may take ONE action per step

Think step by step. For each agent output, check for the hallucination types above.
Be a skeptical but fair reviewer. Not every output is hallucinated."""


def build_user_prompt(observation: dict) -> str:
    task_board = "\n".join([
        f"  [{t['status']:12s}] {t['task_id']}: {t['title']} "
        f"(assigned: {t['assigned_to']})"
        + (f" [BLOCKED by: {', '.join(t['blocked_by'])}]" if t.get('blocked_by') else "")
        for t in observation["task_board"]
    ])

    messages = "\n\n".join([
        f"--- {m['agent']} re: {m['task_id']} (step {m['step']}) ---\n{m['content']}"
        for m in observation["recent_messages"]
    ])

    return f"""=== CURRENT PROJECT STATE (Step {observation['step']}/{observation['max_steps']}) ===

[Task Board]
{task_board}

[Recent Agent Outputs]
{messages}

What is your next action? Respond with exactly ONE action from the available list.
State your reasoning first (1-3 sentences with specific evidence), then on the LAST line output ONLY the action (no quotes around the whole line), e.g. APPROVE(T001) or FLAG(T002, "brief evidence")."""


# ── Dataset Generation ────────────────────────────────────────────────────────

def generate_training_samples(
    difficulty: str,
    num_tasks: int,
    n_samples: int = 200,
    seed_start: int = 0,
) -> list[dict]:
    """
    Generate (prompt, seed_metadata) pairs for GRPO training.

    The seed is embedded in the prompt so the reward function can extract it
    and reconstruct the env deterministically. The seed fully determines the
    episode, so reconstruction is correct.

    NOTE (FIX #24): The seed tag is model-visible. In a production setting,
    store metadata externally and pass via side channel. For this hackathon,
    the tag is needed for stateless reward computation due to TRL limitations.
    """
    samples = []
    for i in range(n_samples):
        episode_seed = seed_start + i
        env = MissionCtrlEnv(
            difficulty = difficulty,
            num_tasks  = num_tasks,
            seed       = episode_seed,
            max_steps  = EPISODE_MAX_STEPS,  # FIX #39: consistent max_steps
        )
        obs, _ = env.reset()

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs)},
        ]

        # Embed seed in a machine-readable tag inside the prompt so the
        # reward function can extract it without relying on TRL kwargs
        prompt[1]["content"] += f"\n\n<!-- seed:{episode_seed}:difficulty:{difficulty}:num_tasks:{num_tasks} -->"

        samples.append({
            "prompt": prompt,
            "episode_seed": str(episode_seed),
        })

    return samples


# ── Reward Function for GRPO ──────────────────────────────────────────────────

def _effective_curriculum() -> list[dict]:
    """
    Full CURRICULUM, or a single short easy phase when MISSIONCTRL_SMOKE_STEPS is set
    (GPU dry run: few GRPO steps, no Hub push in train()).
    """
    raw = os.environ.get("MISSIONCTRL_SMOKE_STEPS", "").strip()
    if not raw:
        return CURRICULUM
    try:
        n = max(1, int(raw))
    except ValueError:
        n = 2
    return [{
        "difficulty":  "easy",
        "num_tasks":   3,
        "steps":       n,
        "min_reward":  0.0,   # always pass gate in smoke; goal is to exercise the loop
        "target":      0.0,
    }]


# ── Model Setup ───────────────────────────────────────────────────────────────

def _device_map_for_load() -> str | None:
    """Optional model-parallel loading; disabled by default for GRPO generation stability."""
    raw = os.environ.get("MISSIONCTRL_DEVICE_MAP", "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return None
    if raw in ("1", "true", "yes", "balanced"):
        return "balanced"
    if raw in ("auto", "sequential"):
        return raw
    return DEVICE_MAP


def _sanitize_model_generation_config(model, tokenizer) -> None:
    """
    Clear inherited `max_length` (often 131072 on hub) so TRL/GRPO's `max_new_tokens`
    does not trigger "Both max_new_tokens and max_length seem to have been set."
    After LoRA, touch wrapper and PEFT `base_model` if they carry their own `generation_config`.
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    candidates: list[object] = [model]
    for attr in ("base_model", "model"):
        b = getattr(model, attr, None)
        if b is not None and b not in candidates:
            candidates.append(b)
    bb = getattr(model, "base_model", None)
    if bb is not None:
        for attr in ("model",):
            inner = getattr(bb, attr, None)
            if inner is not None and inner not in candidates:
                candidates.append(inner)

    for m in candidates:
        g = getattr(m, "generation_config", None)
        if g is None:
            continue
        try:
            gdict = g.to_dict()
        except Exception:
            try:
                g.max_length = None
            except Exception:
                pass
            continue
        gdict["max_length"] = None
        gdict["min_length"] = 0
        if pad is not None:
            gdict["pad_token_id"] = pad
        if eos is not None:
            gdict["eos_token_id"] = eos
        # Let TRL supply max_new_tokens at call time; avoid stale new_tokens on config.
        if "max_new_tokens" in gdict:
            gdict["max_new_tokens"] = None
        try:
            if hasattr(GenerationConfig, "from_dict"):
                m.generation_config = GenerationConfig.from_dict(gdict)
            else:  # pragma: no cover
                m.generation_config = GenerationConfig(**gdict)
        except Exception as e:
            logger.debug("Could not rebuild generation_config: %s", e)
            try:
                m.generation_config.max_length = None
            except Exception:
                pass


def _grpo_config_generation_extras_if_supported(tokenizer) -> dict:
    """
    If this TRL version supports it, pass a minimal GenerationConfig (no max_length) so
    `generate` does not merge hub defaults with max_new_tokens. Omitted on older TRL.

    Not merged into GRPOConfig by default: Transformers 5.5+ warns when both
    ``generation_config`` and call-time kwargs overlap. Opt in with
    MISSIONCTRL_USE_GRPO_GENERATION_CONFIG=1, or legacy MISSIONCTRL_SKIP_GRPO_GENERATION_CONFIG=0.
    """
    try:
        if "generation_config" not in inspect.signature(GRPOConfig.__init__).parameters:
            return {}
    except (TypeError, OSError, ValueError):
        return {}
    return {
        "generation_config": GenerationConfig(
            max_length    = None,
            pad_token_id  = getattr(tokenizer, "pad_token_id", None),
            bos_token_id  = getattr(tokenizer, "bos_token_id", None),
            eos_token_id  = getattr(tokenizer, "eos_token_id", None),
        )
    }


def load_model():
    """Load base model with Unsloth optimizations and LoRA adapters."""
    load_kwargs = dict(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,  # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit   = True,  # QLoRA — fits in 16GB VRAM
    )
    dm = _device_map_for_load()
    if dm is not None:
        load_kwargs["device_map"] = dm
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha           = LORA_RANK * 2,
        lora_dropout         = 0.0,  # 0.05 has measurable throughput cost in Unsloth/PEFT
        bias                 = "none",
        use_gradient_checkpointing = "unsloth",   # ~30% VRAM reduction
        random_state         = 42,
    )

    # FIX #11: set padding_side to "left" for correct batched generation with GRPO
    tokenizer.pad_token   = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _sanitize_model_generation_config(model, tokenizer)

    return model, tokenizer


# ── Reward Curve Plotting ─────────────────────────────────────────────────────

def plot_reward_curve(history: list[dict], output_path: str = "reward_curve.png"):
    """
    Generate the reward progression plot used in the pitch demo.
    Shows per-phase rewards with curriculum phase annotations.
    """
    # FIX #12: guard against empty history
    if not history:
        logger.warning("plot_reward_curve: no history to plot, skipping.")
        return

    phases  = [h["phase"] for h in history]
    rewards = [h["avg_reward"] for h in history]
    labels  = [f"Phase {h['phase']}\n({h['difficulty']})" for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phases, rewards, "o-", linewidth=2.5, markersize=9, color="#1D9E75")
    ax.axhline(y=rewards[0], color="#888780", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Baseline: {rewards[0]:.2f}")

    # FIX #5: add ceiling line at 0.85 (not 1.0)
    ax.axhline(y=0.85, color="#CC4444", linestyle=":", linewidth=1, alpha=0.7,
               label="Theoretical ceiling: 0.85")

    for phase, reward, label in zip(phases, rewards, labels):
        ax.annotate(f"{reward:.2f}", xy=(phase, reward),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#0F6E56")

    ax.set_xticks(phases)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("MissionCtrl — Curriculum Training Reward Progression", fontsize=12, pad=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Reward curve saved → {output_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    difficulty: str = "hard",
    num_tasks: int  = 4,
    n_episodes: int | None = None,
    is_mid_training: bool = False,
) -> tuple[float, dict]:
    """
    Run N evaluation episodes and return (mean_reward, aggregated metrics).
    Uses greedy decoding (do_sample=False) for reproducible eval.  # FIX #23
    """
    if n_episodes is None:
        n_episodes = int(os.environ.get("MISSIONCTRL_EVAL_EPISODES", "10").strip() or "10")
    FastLanguageModel.for_inference(model)

    rewards      = []
    detect_rates = []
    fp_rates     = []

    for ep in range(n_episodes):
        env      = MissionCtrlEnv(
            difficulty=difficulty,
            num_tasks=num_tasks,
            seed=9000 + ep,
            max_steps=EPISODE_MAX_STEPS,  # FIX #39: consistent max_steps
        )
        obs, _   = env.reset()
        done     = False
        steps    = 0

        while not done and steps < EPISODE_MAX_STEPS:
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            mdev = getattr(model, "device", None) or next(model.parameters()).device
            inputs = tokenizer(
                prompt_text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = MAX_SEQ_LEN - COMPLETION_MAX_TOKENS,
            ).to(mdev)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = COMPLETION_MAX_TOKENS,
                    do_sample      = False,   # FIX #23: truly greedy for reproducible eval
                )

            completion = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens = True,
            )
            action = parse_action(completion)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            steps += 1

        # FIX #3: use FINAL reward (single state score), not accumulated sum
        ep_reward = compute_reward(env)
        rewards.append(ep_reward)
        detect_rates.append(info.get("detection_rate", 0.0))
        fp_rates.append(info.get("false_positive_rate", 0.0))

    # FIX #35: always restore training mode after eval (is_mid_training param kept for API compat)
    FastLanguageModel.for_training(model)

    mean_reward = np.mean(rewards)
    metrics = {
        "mean_reward":      round(float(mean_reward), 3),
        "std_reward":       round(float(np.std(rewards)), 3),
        "mean_detect_rate": round(float(np.mean(detect_rates)), 3),
        "mean_fp_rate":     round(float(np.mean(fp_rates)), 3),
    }
    print(
        f"    Eval ({n_episodes} eps, {difficulty}): "
        f"reward={metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f} | "
        f"detect={metrics['mean_detect_rate']:.1%} | "
        f"fp={metrics['mean_fp_rate']:.1%}"
    )
    return mean_reward, metrics


# ── Baseline ──────────────────────────────────────────────────────────────────

def run_baseline() -> float:
    """
    Run untrained baseline: approve-everything strategy.
    Establishes the pre-training reward floor (~0.25–0.35 on hard).
    """
    print("📊 Running pre-training baseline (approve-everything strategy)...")
    rewards = []

    for ep in range(20):
        env = MissionCtrlEnv(
            difficulty="hard",
            num_tasks=4,
            seed=ep,
            max_steps=EPISODE_MAX_STEPS,
        )
        env.reset()

        # Approve all tasks without flagging anything
        from environment import TaskStatus
        done = False
        steps = 0
        while not done and steps < EPISODE_MAX_STEPS:
            # Find first in-progress task and approve it
            approved = False
            for task in env._tasks:
                if task.status == TaskStatus.IN_PROGRESS:
                    action = OverseerAction("APPROVE", task_id=task.task_id)
                    _, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    approved = True
                    break
            if not approved:
                # No in-progress tasks; synthesize to close
                _, _, terminated, truncated, info = env.step(OverseerAction("SYNTHESIZE"))
                done = True
            steps += 1

        # FIX #3/#4: use FINAL reward (not accumulated sum, not overwritten per-task)
        total_reward = compute_reward(env)
        rewards.append(total_reward)
        print(
            f"  Ep {ep:2d}: reward={total_reward:.3f} | "
            f"detected {info['caught_count']}/{info['injected_count']} hallucinations"
        )

    mean = float(np.mean(rewards))
    print(f"\n  Baseline mean reward: {mean:.3f}")
    print(f"  (Expected post-training: 0.68+; ceiling is 0.85)")
    if not _curriculum_gate_enabled():
        print(
            "  ℹ️  This baseline is for reference only; curriculum does not block on phase eval "
            "when MISSIONCTRL_CURRICULUM_GATE is unset or 0."
        )
    return mean


# ── GRPO training callbacks (council: stop if easy-phase reward is flat) ─────

def _extract_log_reward(logs: dict) -> float | None:
    """Best-effort scalar from TRL/HF `on_log` payloads (keys vary by version)."""
    for key in (
        "rewards",
        "reward",
        "train/reward",
        "train/reward_mean",
        "grpo/reward",
    ):
        v = logs.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return float(np.mean(v))
    return None


class FlatRewardEarlyStopCallback(TrainerCallback):
    """
    If easy-phase (curriculum index 0) reward stops moving after `min_step`, end the phase early.
    Opt-in via MISSIONCTRL_EARLY_STOP_PHASE1=1: log-scale GRPO rewards can look "flat" while eval
    is still improving, so this is off by default.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        phase_index: int,
        min_step: int = 75,
        log_window: int = 3,
        flat_delta: float = 0.02,
    ):
        self.enabled = bool(enabled) and phase_index == 0
        self.min_step = min_step
        self.log_window = max(2, log_window)
        self.flat_delta = flat_delta
        self._history: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or logs is None:
            return
        if state.global_step < self.min_step:
            return
        val = _extract_log_reward(logs)
        if val is None:
            return
        self._history.append(val)
        self._history = self._history[-self.log_window :]
        if len(self._history) < self.log_window:
            return
        if max(self._history) - min(self._history) < self.flat_delta:
            control.should_training_stop = True
            logger.warning(
                "Early stop (easy phase): flat reward in last %d logs %s at step %s",
                self.log_window,
                self._history,
                state.global_step,
            )


# ── Training Loop (optional curriculum reward gate) ──────────────────────────

def train():
    is_smoke = bool(os.environ.get("MISSIONCTRL_SMOKE_STEPS", "").strip())
    if is_smoke:
        print(
            f"🔬 MISSIONCTRL_SMOKE_STEPS={os.environ.get('MISSIONCTRL_SMOKE_STEPS', '').strip()}: "
            "single easy phase; push_to_hub disabled at end."
        )
    print("🚀 Loading model...")
    model, tokenizer = load_model()
    # tokenizer.pad_token and padding_side already set in load_model()

    all_rewards_history = []
    curriculum = _effective_curriculum()
    curriculum_gate = _curriculum_gate_enabled()
    if curriculum_gate:
        print(
            "  ℹ️  MISSIONCTRL_CURRICULUM_GATE=1: phases may repeat until eval ≥ min_reward "
            f"or {MAX_PHASE_REPEATS + 1} attempts."
        )
    else:
        print(
            "  ℹ️  Curriculum reward gate off (default): each phase runs once; "
            "post-phase eval is advisory only."
        )

    for phase_idx, phase in enumerate(curriculum):
        base_steps = int(phase["steps"])
        max_attempts = (MAX_PHASE_REPEATS + 1) if curriculum_gate else 1
        phase_attempts = 0

        while True:
            phase_attempts += 1
            if curriculum_gate:
                effective_steps = base_steps if phase_attempts == 1 else int(base_steps * 1.4) + 24
                attempt_label = f"(attempt {phase_attempts}/{max_attempts})"
            else:
                effective_steps = base_steps
                attempt_label = "(single pass)"

            print(f"\n{'=' * 60}")
            print(
                f"📚 Curriculum Phase {phase_idx + 1}/{len(curriculum)}: {phase['difficulty'].upper()} "
                f"| {phase['num_tasks']} tasks | {effective_steps} GRPO steps {attempt_label}"
            )
            print(f"{'=' * 60}")

            print("  Generating training samples...")
            samples = generate_training_samples(
                difficulty  = phase["difficulty"],
                num_tasks   = phase["num_tasks"],
                n_samples   = _phase_n_samples(effective_steps),
                seed_start  = phase_idx * 1000 + phase_attempts * 100,
            )
            dataset = Dataset.from_list(samples)

            early_cb = FlatRewardEarlyStopCallback(
                # Default off: "flat" GRPO log rewards often cut phase-1 short before min_reward eval improves.
                enabled=os.environ.get("MISSIONCTRL_EARLY_STOP_PHASE1", "0").strip().lower()
                in ("1", "true", "yes"),
                phase_index=phase_idx,
                min_step=int(os.environ.get("MISSIONCTRL_EARLY_STOP_MIN_STEPS", "75")),
                log_window=int(os.environ.get("MISSIONCTRL_EARLY_STOP_LOG_WINDOW", "3")),
            )

            _train_temp = float(os.environ.get("MISSIONCTRL_GRPO_TEMPERATURE", "0.55").strip() or "0.55")

            _grpo_kwargs = dict(
                output_dir                  = f"{OUTPUT_DIR}/phase_{phase_idx + 1}_attempt_{phase_attempts}",
                num_train_epochs            = 1,
                max_steps                   = effective_steps,
                per_device_train_batch_size = BATCH_SIZE,
                gradient_accumulation_steps = GRAD_ACCUM,
                learning_rate               = LEARNING_RATE,
                num_generations             = NUM_GENERATIONS,   # FIX #2: ≤ BATCH_SIZE
                max_completion_length       = COMPLETION_MAX_TOKENS,
                max_prompt_length           = MAX_SEQ_LEN - COMPLETION_MAX_TOKENS,  # FIX #22: lockstep with completion budget
                temperature                 = _train_temp,
                logging_steps               = 10,
                save_steps                  = SAVE_STEPS,
                report_to                   = _DEFAULT_REPORT,
                seed                        = 42,
            )
            _use_grpo_gc = os.environ.get(
                "MISSIONCTRL_USE_GRPO_GENERATION_CONFIG", ""
            ).strip().lower() in ("1", "true", "yes")
            _legacy_merge_gc = os.environ.get(
                "MISSIONCTRL_SKIP_GRPO_GENERATION_CONFIG", ""
            ).strip().lower() in ("0", "false", "no")
            if _use_grpo_gc or _legacy_merge_gc:
                _grpo_kwargs.update(_grpo_config_generation_extras_if_supported(tokenizer))
            try:
                grpo_config = GRPOConfig(**_grpo_kwargs)
            except TypeError:
                _grpo_kwargs.pop("generation_config", None)
                grpo_config = GRPOConfig(**_grpo_kwargs)

            trainer = GRPOTrainer(
                model        = model,
                tokenizer    = tokenizer,
                reward_funcs = grpo_reward_fn,
                args         = grpo_config,
                train_dataset = dataset,
                callbacks     = [early_cb],
            )

            trainer.train()

            avg_reward, metrics = evaluate(
                model, tokenizer,
                phase["difficulty"], phase["num_tasks"],
                is_mid_training=True,
            )

            history_entry = {
                "phase":      phase_idx + 1,
                "difficulty": phase["difficulty"],
                "avg_reward": avg_reward,
                "metrics":    metrics,
                "attempts":   phase_attempts,
            }

            if not curriculum_gate:
                all_rewards_history.append(history_entry)
                print(
                    f"\n  Phase {phase_idx + 1} post-train eval (advisory): reward={avg_reward:.3f} | "
                    f"curriculum reference threshold={phase['min_reward']:.2f} (not used to block or repeat)."
                )
                print(
                    "  ℹ️  Logged GRPO reward optimizes the first completion step (+ scripted rollout in "
                    "grpo_rewards); eval is full greedy episodes — they can diverge until the policy generalizes."
                )
                break

            if avg_reward >= phase["min_reward"]:
                all_rewards_history.append(history_entry)
                print(
                    f"\n  ✅ Phase {phase_idx + 1} PASSED | "
                    f"reward={avg_reward:.3f} ≥ threshold={phase['min_reward']:.2f}"
                )
                break
            if phase_attempts >= max_attempts:
                all_rewards_history.append(history_entry)
                print(
                    f"\n  ⚠️  Phase {phase_idx + 1} threshold not met: "
                    f"{avg_reward:.3f} < {phase['min_reward']:.2f} — max attempts reached, advancing."
                )
                break
            print(
                f"\n  ⚠️  Phase {phase_idx + 1} threshold not met: "
                f"{avg_reward:.3f} < {phase['min_reward']:.2f}\n     Repeating phase..."
            )

    # Save reward curve
    print("\n📈 Generating reward curve...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_reward_curve(all_rewards_history, f"{OUTPUT_DIR}/reward_curve.png")

    # Final summary
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE")
    print(f"   Reward ceiling: 0.85 | Target: ≥0.68")  # FIX #5
    print("=" * 60)
    for entry in all_rewards_history:
        pct_of_ceiling = entry['avg_reward'] / 0.85 * 100
        print(
            f"  Phase {entry['phase']} ({entry['difficulty']:6s}): "
            f"{entry['avg_reward']:.3f} ({pct_of_ceiling:.0f}% of ceiling) "
            f"(detect={entry['metrics']['mean_detect_rate']:.1%}, "
            f"fp={entry['metrics']['mean_fp_rate']:.1%})"
        )

    # FIX #28: push LoRA adapter to Hub; local save also uses lora for consistency
    os.makedirs(f"{OUTPUT_DIR}/final_lora", exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    if is_smoke:
        print(f"  ℹ️  Smoke run: skipped push_to_hub (local adapter: {OUTPUT_DIR}/final_lora)")
    else:
        print(f"\n📤 Pushing to HuggingFace Hub: {HF_REPO}")
        model.push_to_hub(HF_REPO, tokenizer=tokenizer)
        print(f"  ✅ LoRA adapter uploaded → https://huggingface.co/{HF_REPO}")
    print(f"  ℹ️  To use: load base model + adapter via PEFT/Unsloth")

    return all_rewards_history


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MissionCtrl Training")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run baseline evaluation only (no training)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval")
    parser.add_argument("--reward-smoke", action="store_true",
                        help="Run grpo_reward_fn on TRL-style completions (no model load)")
    parser.add_argument("--smoke-train", action="store_true",
                        help="Run one easy phase; uses MISSIONCTRL_SMOKE_STEPS (default 2); no Hub push")
    args = parser.parse_args()

    if args.reward_smoke:
        from grpo_rewards import run_reward_smoke
        raise SystemExit(0 if run_reward_smoke() else 1)
    if args.smoke_train:
        if not os.environ.get("MISSIONCTRL_SMOKE_STEPS", "").strip():
            os.environ["MISSIONCTRL_SMOKE_STEPS"] = "2"
        print("🔬 --smoke-train: short GRPO run (no baseline, Hub push disabled)\n")
        train()
    elif args.baseline_only:
        run_baseline()
    elif args.eval_only and args.checkpoint:
        model, tokenizer = load_model()
        model.load_adapter(args.checkpoint)
        evaluate(model, tokenizer)
    else:
        if os.environ.get("MISSIONCTRL_SKIP_BASELINE", "").strip().lower() in (
            "1", "true", "yes",
        ):
            print("  ⏭️  MISSIONCTRL_SKIP_BASELINE set — skipping pre-training baseline")
        else:
            baseline = run_baseline()
            print(f"\n🎯 Baseline established: {baseline:.3f} (ceiling: 0.85)")
        _gate_on = _curriculum_gate_enabled()
        print(
            "  ℹ️  MISSIONCTRL_CURRICULUM_GATE="
            f"{'1 — phases may repeat until eval ≥ min_reward' if _gate_on else '0 — each phase runs once; eval is advisory'}"
        )
        print("Starting curriculum training...\n")
        train()