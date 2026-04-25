"""Parity tests between root `environment` (GRPO) and `server.environment` (HTTP API).

The two engines use different state machines and task banks; this file locks
invariants that must not diverge: overseer action parsing and composite weight
constants used for documentation alignment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `environment.py` expects `openenv.Env`; some `openenv` metapackage builds omit it.
import openenv as _openenv

try:
    _ = _openenv.Env
except AttributeError:
    class _StubEnv:  # noqa: D401
        metadata = {}

    _openenv.Env = _StubEnv  # type: ignore[attr-defined]

import environment as root_env
import server.environment as server_env


def _canonical(t: str) -> str:
    if t in ("SYNTHESIZE", "SYNTHESIZE_REPORT"):
        return "SYNTHESIZE"
    return t


SAMPLES = [
    "APPROVE(task_01)",
    'REJECT(task_02, "scope issue")',
    "REDELEGATE(task_03, CoderAgent)",
    "FLAG(task_04, fabricated citation in paragraph 2)",
    "ESCALATE(task_05)",
    "SYNTHESIZE_REPORT()",
    "NOOP",
]


class TestParseActionParity:
    def test_shared_canonical_actions(self):
        for s in SAMPLES:
            r = root_env.parse_action(s)
            p = server_env.parse_action(s)
            assert _canonical(r.action_type) == _canonical(
                p.action_type
            ), f"action type mismatch for {s!r}: {r.action_type!r} vs {p.action_type!r}"

    def test_root_alternate_synthesize_form(self):
        r = root_env.parse_action("SYNTHESIZE()")
        p = server_env.parse_action("SYNTHESIZE()")
        assert r.action_type == "SYNTHESIZE"
        assert p.action_type == "NOOP"

    def test_server_json_fallback(self):
        s = '{"action_type": "APPROVE", "task_id": "task_99"}'
        p = server_env.parse_action(s)
        assert p.action_type == "APPROVE"
        assert p.task_id == "task_99"
        r = root_env.parse_action(s)
        assert r.action_type == "NOOP"

    def test_five_signal_weights_in_server_engine(self):
        eng = server_env.MissionCtrlEngine()
        eng.reset("easy", seed=1)

        raw = eng._compute_raw_reward()
        assert 0.0 < raw < 1.0
        s1, s2, s3, s4, s5 = (
            eng._signal_task_completion(),
            eng._signal_hallucination_detection(),
            eng._signal_false_positive_penalty(),
            eng._signal_delegation_efficiency(),
            eng._signal_llm_judge_mock(),
        )
        expected = 0.30 * s1 + 0.30 * s2 - 0.15 * s3 + 0.15 * s4 + 0.10 * s5
        assert abs(expected - raw) < 1e-5


class TestTaskTierCoverage:
    def test_difficulty_tiers_match_set(self):
        from server.environment import DIFFICULTY_CONFIG

        assert set(DIFFICULTY_CONFIG) == {"easy", "medium", "hard", "special"}
