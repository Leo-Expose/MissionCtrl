"""reward_model.compute_reward open-interval parity with server grader."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import MissionCtrlEnv, OverseerAction
from reward_model import _SCORE_EPS, _clamp_open_interval, compute_reward


class TestOpenIntervalClamp:
    def test_clamp_edges(self):
        assert _clamp_open_interval(0.0) == _SCORE_EPS
        assert _clamp_open_interval(1.0) == 1.0 - _SCORE_EPS
        assert _clamp_open_interval(-9.0) == _SCORE_EPS
        assert _clamp_open_interval(9.0) == 1.0 - _SCORE_EPS
        assert _clamp_open_interval(0.5) == 0.5

    def test_compute_reward_strictly_between_zero_and_one(self):
        for difficulty in ("easy", "medium", "hard", "special"):
            for seed in (0, 7, 42, 99):
                env = MissionCtrlEnv(
                    difficulty=difficulty,
                    num_tasks=3,
                    max_steps=8,
                    seed=seed,
                )
                env.reset()
                for _ in range(8):
                    r = compute_reward(env)
                    assert 0.0 < r < 1.0, (
                        f"compute_reward={r} not in (0,1) for {difficulty} seed={seed}"
                    )
                    env.step(OverseerAction("NOOP"))
