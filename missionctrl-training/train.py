"""
MissionCtrl Training Script
============================
GRPO fine-tuning with Unsloth on the MissionCtrl environment.
Runs in Google Colab with the provided HuggingFace compute credits.

Requirements (run this cell first in Colab):
  !pip install unsloth trl openenv transformers datasets accelerate matplotlib
  !pip install --upgrade bitsandbytes

Model  : Qwen2.5-7B-Instruct (fast to train, strong reasoning baseline)
Method : GRPO (Group Relative Policy Optimization) via TRL

Expected training time on A100: ~2-3 hours for visible reward improvement
Expected reward curve: 0.28 → 0.75+ after 500 steps
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import random
import re
import torch
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

# Local env
sys.path.insert(0, os.path.dirname(__file__))
from environment import MissionCtrlEnv, OverseerAction, parse_action
from reward_model import compute_reward, reward_breakdown

# Colab-specific: mount Google Drive for checkpoint persistence
try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = "/content/drive/MyDrive/missionctrl_checkpoints"
except ImportError:
    OUTPUT_DIR = "./missionctrl_checkpoints"

# Kaggle-specific: load HF_TOKEN from secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except ImportError:
        pass

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"   # swap for Llama-3.1-8B-Instruct if preferred
MAX_SEQ_LEN     = 4096
LORA_RANK       = 16
BATCH_SIZE      = 4
GRAD_ACCUM      = 4                             # effective batch = 16
LEARNING_RATE   = 2e-5
NUM_GENERATIONS = 4                             # GRPO: samples per prompt
SAVE_STEPS      = 50
HF_REPO         = "Proliferation/missionctrl"  # set before push
RESUME_FROM_CHECKPOINT = None  # Set to path to resume training (e.g., "./missionctrl_checkpoints/phase_1_attempt_1/checkpoint-100")

# Validate HF_REPO is not placeholder
assert "your-hf" not in HF_REPO, "Please set your actual HF username in HF_REPO"

# Platform-specific configs
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_count() >= 2:
        # Kaggle T4 x2
        DEVICE_MAP = "balanced"
        BATCH_SIZE = 2
        NUM_GENERATIONS = 4
        GRAD_ACCUM = 8
    else:
        # Colab A100 or single GPU
        DEVICE_MAP = None
        BATCH_SIZE = 4
        NUM_GENERATIONS = 8
        GRAD_ACCUM = 4
except:
    DEVICE_MAP = None
    BATCH_SIZE = 4
    NUM_GENERATIONS = 8
    GRAD_ACCUM = 4

# Curriculum: start easy, escalate — gate on reward threshold before advancing
# Platform-specific: Kaggle gets reduced steps for 9-hour session limit
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_count() >= 2:
        # Kaggle: reduced steps for 9-hour session
        CURRICULUM = [
            {"difficulty": "easy",   "num_tasks": 2, "steps": 100, "min_reward": 0.50, "target": 0.55},
            {"difficulty": "medium", "num_tasks": 3, "steps": 150, "min_reward": 0.62, "target": 0.68},
            {"difficulty": "hard",   "num_tasks": 4, "steps": 100, "min_reward": 0.75, "target": 0.80},
        ]
    else:
        # Colab: full curriculum
        CURRICULUM = [
            {"difficulty": "easy",   "num_tasks": 2, "steps": 150, "min_reward": 0.50, "target": 0.55},
            {"difficulty": "medium", "num_tasks": 3, "steps": 200, "min_reward": 0.62, "target": 0.68},
            {"difficulty": "hard",   "num_tasks": 4, "steps": 150, "min_reward": 0.75, "target": 0.80},
        ]
except:
    # Fallback: full curriculum
    CURRICULUM = [
        {"difficulty": "easy",   "num_tasks": 2, "steps": 150, "min_reward": 0.50, "target": 0.55},
        {"difficulty": "medium", "num_tasks": 3, "steps": 200, "min_reward": 0.62, "target": 0.68},
        {"difficulty": "hard",   "num_tasks": 4, "steps": 150, "min_reward": 0.75, "target": 0.80},
    ]
MAX_PHASE_REPEATS = 2   # repeat a phase up to this many times if threshold not met
EARLY_STOP_PATIENCE = 3  # stop if reward doesn't improve for N evaluations

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Oversight Agent (OverseerAgent) responsible for managing a fleet of 5 specialist sub-agents working on a software engineering project.

Your sub-agents are:
- PlannerAgent:    Decomposes tasks and sets milestones
- ResearchAgent:   Gathers context and domain knowledge
- CoderAgent:      Writes implementation code and tests
- TesterAgent:     Verifies outputs and runs test suites
- CommAgent:       Drafts reports and stakeholder communications

CRITICAL: Some sub-agent outputs contain hallucinations — fabricated citations, non-existent APIs, circular reasoning, false metrics, scope creep, or lazy self-delegation. You must detect and flag them.

Available actions (ONE per step):
- APPROVE(task_id)                          — accept a clean, correct output
- REJECT(task_id, "reason")                 — reject with explanation
- REDELEGATE(task_id, AgentName)            — reassign task to a different agent
- FLAG(task_id, "evidence")   — flag a corrupted output with specific evidence
- ESCALATE(task_id)                         — mark task as blocked
- SYNTHESIZE_REPORT()                       — compile final output (only when all hallucinations caught)

Rules:
1. Always provide SPECIFIC evidence when flagging — quote or reference exact content from the output
2. Be precise — false positives (flagging clean outputs) are penalized
3. When rejecting, always give a clear reason
4. You may take ONE action per step

Think step by step. For each agent output, check for:
- Fabricated or unverifiable sources/citations
- Impossible API calls or non-existent libraries
- Contradictions with previously approved outputs
- Circular or unsupported reasoning
- Implausibly perfect or unverifiable metrics
- Scope violations or unauthorized task expansion
- Self-delegation under a different name

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
State your reasoning first (1-3 sentences with specific evidence), then your action on the final line."""


# ── Dataset Generation ────────────────────────────────────────────────────────

def generate_training_samples(
    difficulty: str,
    num_tasks: int,
    n_samples: int = 300,
    seed_start: int = 0,
) -> list[dict]:
    """
    Generate (prompt, seed_metadata) pairs for GRPO training.

    FIXED: env_state is not passed via kwargs to the reward fn (TRL limitation).
    Instead we embed the seed in the prompt and reconstruct env from it in the
    reward function. The seed fully determines the episode, so reconstruction
    is deterministic and correct.
    """
    samples = []
    for i in range(n_samples):
        episode_seed = seed_start + i
        env = MissionCtrlEnv(
            difficulty = difficulty,
            num_tasks  = num_tasks,
            seed       = episode_seed,
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
            # Also store as a top-level field for any framework that surfaces it
            "episode_seed": str(episode_seed),
        })

    return samples


# ── Reward Function for GRPO ──────────────────────────────────────────────────

def grpo_reward_fn(completions: list[str], prompts: list, **kwargs) -> list[float]:
    """
    GRPO reward function called by TRL.

    LIMITATION: GRPO currently only supports single-step RL. To work around this,
    we run the entire episode step-by-step from the seed and evaluate the final
    outcome. The three-phase curriculum partially compensates for this limitation
    by gradually increasing the sequence length and difficulty.

    FIXED: extracts episode seed from the embedded tag in the prompt text,
    reconstructs the environment deterministically, applies the model's action,
    and returns the reward. This is fully stateless and avoids the TRL kwargs
    limitation that prevented env_state from being passed.
    """
    rewards = []

    for i, completion in enumerate(completions):
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
                episode_seed = i
                difficulty   = "medium"
                num_tasks    = 3

            # Reconstruct env deterministically from seed
            env = MissionCtrlEnv(
                difficulty = difficulty,
                num_tasks  = num_tasks,
                seed       = episode_seed,
            )
            env.reset(seed=episode_seed)

            # Parse and apply the model's action
            action = parse_action(completion)
            _, reward, _, _, _ = env.step(action)
            rewards.append(float(reward))

        except Exception as e:
            # Malformed completion or env error → zero reward
            rewards.append(0.0)

    return rewards


# ── Model Setup ───────────────────────────────────────────────────────────────

def load_model():
    """Load base model with Unsloth optimizations and LoRA adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,         # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit   = True,         # QLoRA — fits in 16GB VRAM
        device_map     = DEVICE_MAP,   # balanced for multi-GPU (Kaggle T4 x2)
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha           = LORA_RANK * 2,
        lora_dropout         = 0.05,
        bias                 = "none",
        use_gradient_checkpointing = "unsloth",   # ~30% VRAM reduction
        random_state         = 42,
    )

    return model, tokenizer


# ── Reward Curve Plotting ─────────────────────────────────────────────────────

def plot_reward_curve(history: list[dict], output_path: str = "reward_curve.png"):
    """
    Generate the reward progression plot used in the pitch demo.
    Shows per-phase rewards with curriculum phase annotations.
    """
    phases  = [h["phase"] for h in history]
    rewards = [h["avg_reward"] for h in history]
    labels  = [f"Phase {h['phase']}\n({h['difficulty']})" for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phases, rewards, "o-", linewidth=2.5, markersize=9, color="#1D9E75")
    ax.axhline(y=rewards[0], color="#888780", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Baseline: {rewards[0]:.2f}")

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
    n_episodes: int = 20,
    is_mid_training: bool = False,
) -> tuple[float, dict]:
    """
    Run N evaluation episodes and return (mean_reward, aggregated_metrics).
    Uses greedy-ish decoding (temperature=0.1) for reproducible eval.
    """
    FastLanguageModel.for_inference(model)

    rewards      = []
    detect_rates = []
    fp_rates     = []

    for ep in range(n_episodes):
        env      = MissionCtrlEnv(difficulty=difficulty, num_tasks=num_tasks, seed=9000 + ep)
        obs, _   = env.reset()
        done     = False
        ep_reward = 0.0
        steps    = 0

        while not done and steps < env.max_steps:
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = MAX_SEQ_LEN - 512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = 256,
                    temperature    = 0.1,
                    do_sample      = True,
                )

            completion = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens = True,
            )
            action = parse_action(completion)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done      = terminated or truncated
            steps    += 1

        rewards.append(ep_reward)
        detect_rates.append(info.get("detection_rate", 0.0))
        fp_rates.append(info.get("false_positive_rate", 0.0))

    if is_mid_training:
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
        env = MissionCtrlEnv(difficulty="hard", num_tasks=4, seed=ep)
        env.reset()

        # Approve all tasks without flagging anything
        total_reward = 0.0
        for task in env._tasks:
            action = OverseerAction("APPROVE", task_id=task.task_id)
            _, reward, _, _, info = env.step(action)
            total_reward = reward

        rewards.append(total_reward)
        print(
            f"  Ep {ep:2d}: reward={total_reward:.3f} | "
            f"detected {info['caught_count']}/{info['injected_count']} hallucinations"
        )

    mean = float(np.mean(rewards))
    print(f"\n  Baseline mean reward: {mean:.3f}")
    print(f"  (Expected post-training: 0.75+)")
    return mean


# ── Training Loop with Curriculum Gating ─────────────────────────────────────

def train():
    # Detect platform for logging
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.get_device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🖥️  Platform: {'Kaggle' if gpu_count >= 2 else 'Colab/Local'}")
            print(f"🎮 GPU: {gpu_name} x{gpu_count}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except:
        print("🖥️  Platform: Unknown")

    print("🚀 Loading model...")
    model, tokenizer = load_model()
    tokenizer.pad_token = tokenizer.eos_token

    all_rewards_history = []
    best_reward = 0.0
    patience_counter = 0

    for phase_idx, phase in enumerate(CURRICULUM):
        phase_attempts = 0
        phase_passed   = False

        while phase_attempts <= MAX_PHASE_REPEATS and not phase_passed:
            phase_attempts += 1
            attempt_label = f"(attempt {phase_attempts}/{MAX_PHASE_REPEATS + 1})"

            print(f"\n{'=' * 60}")
            print(
                f"📚 Curriculum Phase {phase_idx + 1}/3: {phase['difficulty'].upper()} "
                f"| {phase['num_tasks']} tasks | {phase['steps']} steps {attempt_label}"
            )
            print(f"{'=' * 60}")

            print("  Generating training samples...")
            samples = generate_training_samples(
                difficulty  = phase["difficulty"],
                num_tasks   = phase["num_tasks"],
                n_samples   = 300,
                seed_start  = phase_idx * 1000 + phase_attempts * 100,
            )
            dataset = Dataset.from_list(samples)

            grpo_config = GRPOConfig(
                output_dir                  = f"{OUTPUT_DIR}/phase_{phase_idx + 1}_attempt_{phase_attempts}",
                num_train_epochs            = 1,
                max_steps                   = phase["steps"],
                per_device_train_batch_size = BATCH_SIZE,
                gradient_accumulation_steps = GRAD_ACCUM,
                learning_rate               = LEARNING_RATE,
                num_generations             = NUM_GENERATIONS,
                max_completion_length       = 512,
                temperature                 = 0.7,
                logging_steps               = 10,
                save_steps                  = SAVE_STEPS,
                report_to                   = "tensorboard",
                seed                        = 42,
                max_grad_norm               = 1.0,  # Gradient clipping for stability
            )

            trainer = GRPOTrainer(
                model        = model,
                tokenizer    = tokenizer,
                reward_funcs = grpo_reward_fn,
                args         = grpo_config,
                train_dataset = dataset,
            )

            trainer.train()

            # Evaluate after training — gate advancement on hitting min_reward
            avg_reward, metrics = evaluate(
                model, tokenizer,
                phase["difficulty"], phase["num_tasks"],
                n_episodes = 20,
                is_mid_training = True,
            )

            if avg_reward >= phase["min_reward"]:
                phase_passed = True
                all_rewards_history.append({
                    "phase":      phase_idx + 1,
                    "difficulty": phase["difficulty"],
                    "avg_reward": avg_reward,
                    "metrics":    metrics,
                    "attempts":   phase_attempts,
                })
                print(
                    f"\n  ✅ Phase {phase_idx + 1} PASSED | "
                    f"reward={avg_reward:.3f} ≥ threshold={phase['min_reward']:.2f}"
                )
                # Reset patience on success
                patience_counter = 0
            else:
                print(
                    f"\n  ⚠️  Phase {phase_idx + 1} threshold not met: "
                    f"{avg_reward:.3f} < {phase['min_reward']:.2f}"
                )
                # Early stopping check
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"     Reward hasn't improved for {patience_counter} evaluations")
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"     🛑 Early stopping triggered (patience={EARLY_STOP_PATIENCE})")
                        all_rewards_history.append({
                            "phase":      phase_idx + 1,
                            "difficulty": phase["difficulty"],
                            "avg_reward": avg_reward,
                            "metrics":    metrics,
                            "attempts":   phase_attempts,
                        })
                        break
                if phase_attempts <= MAX_PHASE_REPEATS:
                    print(f"     Repeating phase...")
                else:
                    print(f"     Max attempts reached — advancing anyway.")
                    all_rewards_history.append({
                        "phase":      phase_idx + 1,
                        "difficulty": phase["difficulty"],
                        "avg_reward": avg_reward,
                        "metrics":    metrics,
                        "attempts":   phase_attempts,
                    })

    # Save reward curve
    print("\n📈 Generating reward curve...")
    plot_reward_curve(all_rewards_history, f"{OUTPUT_DIR}/reward_curve.png")

    # Final summary
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE")
    print("=" * 60)
    for entry in all_rewards_history:
        print(
            f"  Phase {entry['phase']} ({entry['difficulty']:6s}): "
            f"{entry['avg_reward']:.3f} "
            f"(detect={entry['metrics']['mean_detect_rate']:.1%}, "
            f"fp={entry['metrics']['mean_fp_rate']:.1%})"
        )

    # Push to HuggingFace Hub
    print(f"\n📤 Pushing to HuggingFace Hub: {HF_REPO}")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/final",
        tokenizer,
        save_method = "merged_16bit",
    )
    model.push_to_hub_merged(HF_REPO, tokenizer, save_method="lora")
    print(f"  ✅ Model uploaded → https://huggingface.co/{HF_REPO}")

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
    args = parser.parse_args()

    if args.baseline_only:
        run_baseline()
    elif args.eval_only and args.checkpoint:
        model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint)
        evaluate(model, tokenizer)
    else:
        baseline = run_baseline()
        print(f"\n🎯 Baseline established: {baseline:.3f}")
        print("Starting curriculum training...\n")
        train()
