"""
GRPO reward rollout for MissionCtrl (no torch) — used by `train.GRPOTrainer` and
`python train.py --reward-smoke`. Keeps heavy imports out of the reward-smoke path.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from grpo_completion import _completion_to_text
from environment import MissionCtrlEnv, OverseerAction, parse_action, TaskStatus
from reward_model import compute_reward

logger = logging.getLogger("train")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Must match train.EPISODE_MAX_STEPS and MissionCtrlEnv default max_steps
EPISODE_MAX_STEPS = 40


def _greedy_completion_action(env: MissionCtrlEnv) -> OverseerAction:
    """
    Simple completion policy used to finish episodes after the model's first action.
    Priority: FLAG uncaught hallucinations → APPROVE clean in-progress tasks → SYNTHESIZE.
    """
    uncaught = env._injected_ids - env._caught_ids
    for task_id in uncaught:
        task = env._get_task(task_id)
        if task and task.status == TaskStatus.IN_PROGRESS:
            return OverseerAction(
                "FLAG",
                task_id=task_id,
                evidence=f"hallucination detected: {task.hallucination_type}"
            )

    for task in env._tasks:
        if task.status == TaskStatus.IN_PROGRESS and task.task_id not in env._injected_ids:
            return OverseerAction("APPROVE", task_id=task.task_id)

    uncaught2 = env._injected_ids - env._caught_ids
    if not uncaught2:
        return OverseerAction("SYNTHESIZE")

    for task in env._tasks:
        if task.status == TaskStatus.IN_PROGRESS:
            return OverseerAction("APPROVE", task_id=task.task_id)

    return OverseerAction("NOOP")


def _single_completion_reward(i: int, completion: Any, prompts: list) -> float:
    """
    One GRPO completion → scalar reward. Used (possibly in parallel) by grpo_reward_fn;
    each call owns a fresh MissionCtrlEnv — no cross-thread state.
    """
    try:
        # Extract seed from embedded tag in prompt
        prompt_text = ""
        if i < len(prompts):
            p = prompts[i]
            if isinstance(p, list):
                prompt_text = " ".join(
                    msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    for msg in p
                )
            else:
                prompt_text = str(p)

        seed_match = re.search(
            r"<!-- seed:(\d+):difficulty:(\w+):num_tasks:(\d+) -->",
            prompt_text
        )

        if seed_match:
            episode_seed = int(seed_match.group(1))
            difficulty   = seed_match.group(2)
            num_tasks    = int(seed_match.group(3))
        else:
            # Fallback defaults if tag not found
            logger.warning("Seed tag not found in prompt %d, using defaults", i)
            episode_seed = i
            difficulty   = "medium"
            num_tasks    = 3

        # Reconstruct env deterministically from seed
        env = MissionCtrlEnv(
            difficulty=difficulty,
            num_tasks=num_tasks,
            seed=episode_seed,
            max_steps=EPISODE_MAX_STEPS,  # FIX #39
        )
        env.reset(seed=episode_seed)

        # FIX #1: Apply model's action for this step, then run episode to completion
        # using a greedy policy (approve first available in-progress task)
        action = parse_action(_completion_to_text(completion))
        _, _, terminated, truncated, _ = env.step(action)

        # Complete the episode with simple greedy policy
        step_count = 1
        while not (terminated or truncated) and step_count < EPISODE_MAX_STEPS:
            greedy_action = _greedy_completion_action(env)
            _, _, terminated, truncated, _ = env.step(greedy_action)
            step_count += 1

        # FIX #3: Use FINAL reward (state score at episode end), not a sum
        final_reward = compute_reward(env)
        return float(final_reward)

    except Exception as e:
        # FIX #20: log the exception so bugs are visible during training
        logger.warning("Reward fn error for completion %d: %s", i, e)
        return 0.0


def grpo_reward_fn(completions: list[Any], prompts: list, **kwargs) -> list[float]:
    """
    GRPO reward function called by TRL.

    FIX #1: Now runs a FULL episode rollout (not just one step).
    The first action comes from the model completion; subsequent actions use a
    greedy policy (approve-all) to complete the episode. This gives a meaningful
    multi-step reward signal rather than a single-step snapshot. (A fuller
    multi-step model rollout in the reward would reduce greedy-completion bias
    but requires a larger TRL integration change.)

    FIX #3: Returns the FINAL reward (single composite score at episode end),
    not a sum of per-step rewards. compute_reward returns a state score in the
    strict open interval (0, 1) (same ε-clamp as the HTTP grader), and summing
    state scores across steps is meaningless.

    FIX #20: Exceptions are now logged (not silently swallowed).

    Coerce `completion` to str (see grpo_completion._completion_to_text) — TRL may
    pass list/dict message parts, unlike evaluate() which decodes a single string.
    """
    n = len(completions)
    if n == 0:
        return []

    raw = os.environ.get("MISSIONCTRL_REWARD_THREADS", "").strip()
    if raw == "1":
        return [_single_completion_reward(i, c, prompts) for i, c in enumerate(completions)]
    if raw:
        try:
            max_workers = max(1, int(raw, 10))
        except ValueError:
            max_workers = min(4, n) or 1
        max_workers = min(max_workers, n)
    else:
        max_workers = min(4, n) or 1
    if max_workers == 1 or n == 1:
        return [_single_completion_reward(i, c, prompts) for i, c in enumerate(completions)]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_single_completion_reward, i, completions[i], prompts) for i in range(n)
        ]
        return [f.result() for f in futures]


def run_reward_smoke() -> bool:
    """
    No GPU: call grpo_reward_fn with str and list-shaped TRL-style completions;
    fails if all rewards are zero (e.g. strip on list bug).
    """
    tag = "<!-- seed:0:difficulty:easy:num_tasks:3 -->"
    prompt = [
        {"role": "system", "content": "x"},
        {"role": "user",   "content": f"u\n\n{tag}"},
    ]
    prompts = [prompt, prompt, prompt, prompt]
    completions = [
        "Step.\nAPPROVE(T001)",
        [{"type": "text", "text": "APPROVE(T001)"}],
        [{"role": "assistant", "content": "APPROVE(T001)"}],
        [{"content": [{"type": "text", "text": "APPROVE(T001)"}]}],
    ]
    rewards = grpo_reward_fn(completions, prompts)
    if len(rewards) != len(completions):
        print(f"❌ reward-smoke: expected {len(completions)} rewards, got {len(rewards)}")
        return False
    if all(r == 0.0 for r in rewards):
        print("❌ reward-smoke: all rewards zero (completion coercion or env bug)")
        return False
    print(f"✅ reward-smoke ok: {len(rewards)} rewards, sample values [{rewards[0]:.3f}, {rewards[1]:.3f}, ...]")
    return True
