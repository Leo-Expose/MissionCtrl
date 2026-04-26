"""Tests for GRPO completion coercion and reward path (no torch required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo_completion import _completion_to_text
from grpo_rewards import grpo_reward_fn, run_reward_smoke


class TestCompletionToText:
    def test_plain_str(self):
        assert _completion_to_text("  APPROVE(T001)  ").strip() == "APPROVE(T001)"

    def test_list_of_text_dicts(self):
        s = _completion_to_text([{"type": "text", "text": "FLAG(T002, \"x\")"}])
        assert "FLAG(T002" in s

    def test_list_of_assistant_message(self):
        s = _completion_to_text([{"role": "assistant", "content": "APPROVE(T001)"}])
        assert "APPROVE(T001)" in s

    def test_nested_multimodal_content(self):
        s = _completion_to_text(
            [{"content": [{"type": "text", "text": "SYNTHESIZE_REPORT()"}]}]
        )
        assert "SYNTHESIZE" in s

    def test_list_of_str_joins(self):
        s = _completion_to_text(["a", "b"])
        assert "a" in s and "b" in s


class TestGrpoRewardFn:
    def test_list_completions_no_strip_error(self):
        tag = "<!-- seed:0:difficulty:easy:num_tasks:3 -->"
        prompt = [
            {"role": "system", "content": "x"},
            {"role": "user", "content": f"u\n\n{tag}"},
        ]
        completions = [
            [{"type": "text", "text": "APPROVE(T001)"}],
        ]
        rewards = grpo_reward_fn(completions, [prompt])
        assert len(rewards) == 1
        assert rewards[0] != 0.0, "coerced list completion should produce non-zero reward"

    def test_run_reward_smoke_succeeds(self):
        assert run_reward_smoke() is True

    def test_parallel_rewards_match_sequential(self):
        tag = "<!-- seed:1:difficulty:easy:num_tasks:3 -->"
        prompt = [
            {"role": "system", "content": "x"},
            {"role": "user",   "content": f"u\n\n{tag}"},
        ]
        prompts = [prompt, prompt, prompt, prompt]
        completions = [
            "APPROVE(T001)",
            "FLAG(T002, \"e\")",
            "APPROVE(T001)",
            "REJECT(T003, \"r\")",
        ]
        os.environ["MISSIONCTRL_REWARD_THREADS"] = "1"
        seq = grpo_reward_fn(completions, prompts)
        os.environ["MISSIONCTRL_REWARD_THREADS"] = "4"
        par = grpo_reward_fn(completions, prompts)
        os.environ.pop("MISSIONCTRL_REWARD_THREADS", None)
        assert len(seq) == len(par) == len(completions)
        for a, b in zip(seq, par, strict=True):
            assert a == b
