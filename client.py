"""Client library for MissionCtrl environment.

Provides HTTP client functions for interacting with the MissionCtrl environment API.
Contains environment payload structures and example scenarios.
"""

import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file automatically
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Known agent types in the environment
KNOWN_AGENTS = (
    "PlannerAgent",
    "ResearchAgent",
    "CoderAgent",
    "TesterAgent",
    "CommAgent",
)

# Available task tiers
TASKS = ["easy", "medium", "hard", "special"]

# Default max steps per episode
MAX_STEPS = 5


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------
http = httpx.Client(timeout=60.0)


# ---------------------------------------------------------------------------
# Environment API Functions
# ---------------------------------------------------------------------------
def reset_env(task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Reset the environment for a specific task.
    
    Args:
        task_id: The task tier to run (easy, medium, hard, special)
        seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary containing the initial observation
    """
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    
    resp = http.post(f"{ENV_BASE_URL}/reset", json=payload)
    resp.raise_for_status()
    return resp.json()


def step_env(action: str) -> Dict[str, Any]:
    """Execute one action in the environment.
    
    Args:
        action: The action string to execute (e.g., "APPROVE(task_1)")
        
    Returns:
        Dictionary containing the new observation, reward, done flag, and info
    """
    resp = http.post(f"{ENV_BASE_URL}/step", json={"action": action})
    resp.raise_for_status()
    return resp.json()


def get_state() -> Dict[str, Any]:
    """Get the current environment state (read-only).
    
    Returns:
        Dictionary containing the current observation
    """
    resp = http.get(f"{ENV_BASE_URL}/state")
    resp.raise_for_status()
    return resp.json()


def get_history() -> list:
    """Get the action history for the current episode.
    
    Returns:
        List of past actions and their results
    """
    resp = http.get(f"{ENV_BASE_URL}/history")
    resp.raise_for_status()
    return resp.json()


def record_result(tier: str, score: float, steps: int, history: list, 
                 score_breakdown: Optional[Dict] = None, 
                 hallucination_stats: Optional[Dict] = None) -> Dict[str, str]:
    """Push a completed episode result to the dashboard.
    
    Args:
        tier: Task tier (easy, medium, hard, special)
        score: Final score for the episode
        steps: Number of steps taken
        history: Action history for the episode
        score_breakdown: Optional detailed score breakdown
        hallucination_stats: Optional hallucination detection statistics
        
    Returns:
        Confirmation response
    """
    payload = {
        "tier": tier,
        "score": score,
        "steps": steps,
        "history": history,
        "score_breakdown": score_breakdown or {},
        "hallucination_stats": hallucination_stats or {},
    }
    resp = http.post(f"{ENV_BASE_URL}/record", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
def example_basic_usage():
    """Example showing basic environment interaction."""
    # Reset environment for easy task
    result = reset_env("easy")
    obs = result["observation"]
    print(f"Started task with {len(obs['tasks'])} tasks")
    
    # Take a step
    step_result = step_env("NOOP")
    print(f"Reward: {step_result['reward']}, Done: {step_result['done']}")
    
    # Get current state
    state = get_state()
    print(f"Current step: {state['time_step']}")


if __name__ == "__main__":
    # For backward compatibility, delegate to inference script
    from inference import main
    main()
