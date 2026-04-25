"""Engine tests for MissionCtrl environment.

Covers: reset, action mechanics, hallucination injection, deterministic replay,
score clamping, dependency graph, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import (
    MissionCtrlEngine, MissionCtrlEnvironment,
    parse_action, _clamp_score, DIFFICULTY_CONFIG, TASK_TIERS,
    AGENTS, HALLUCINATION_TYPES,
)


class TestScoreClamping:
    """All scores must be strictly in (0, 1) — never 0.0 or 1.0."""

    def test_clamp_zero(self):
        assert _clamp_score(0.0) == 0.01

    def test_clamp_one(self):
        assert _clamp_score(1.0) == 0.99

    def test_clamp_negative(self):
        assert _clamp_score(-5.0) == 0.01

    def test_clamp_above_one(self):
        assert _clamp_score(1.5) == 0.99

    def test_clamp_normal(self):
        assert _clamp_score(0.5) == 0.5

    def test_clamp_boundary_epsilon(self):
        result = _clamp_score(0.01)
        assert 0.0 < result < 1.0

    def test_all_tasks_produce_valid_scores(self):
        """Every task tier must produce a score strictly in (0, 1)."""
        for task_id in ["easy", "medium", "hard", "special"]:
            env = MissionCtrlEngine()
            env.reset(task_id, seed=42)
            # Run a few NOOPs then grade
            for _ in range(3):
                env.step("NOOP")
            env.done = True
            score = env.grade()
            assert 0.0 < score < 1.0, f"Score for {task_id} out of range: {score}"


class TestActionParser:
    """Action parser must handle all formats with NOOP fallback."""

    def test_approve(self):
        p = parse_action("APPROVE(task_01)")
        assert p.action_type == "APPROVE"
        assert p.task_id == "task_01"

    def test_reject(self):
        p = parse_action('REJECT(task_02, "low quality output")')
        assert p.action_type == "REJECT"
        assert p.task_id == "task_02"
        assert "low quality" in p.reason

    def test_redelegate(self):
        p = parse_action("REDELEGATE(task_03, CoderAgent)")
        assert p.action_type == "REDELEGATE"
        assert p.task_id == "task_03"
        assert p.agent == "CoderAgent"

    def test_flag(self):
        p = parse_action('FLAG(task_04, "fabricated citation detected")')
        assert p.action_type == "FLAG"
        assert p.task_id == "task_04"
        assert "fabricated" in p.evidence

    def test_escalate(self):
        p = parse_action("ESCALATE(task_05)")
        assert p.action_type == "ESCALATE"
        assert p.task_id == "task_05"

    def test_synthesize(self):
        p = parse_action("SYNTHESIZE_REPORT()")
        assert p.action_type == "SYNTHESIZE_REPORT"

    def test_malformed_falls_back_to_noop(self):
        """CRITICAL: malformed input must NOT become SYNTHESIZE_REPORT."""
        p = parse_action("garbage input blah blah")
        assert p.action_type == "NOOP"

    def test_empty_string_noop(self):
        p = parse_action("")
        assert p.action_type == "NOOP"

    def test_json_format(self):
        p = parse_action('{"action_type": "APPROVE", "task_id": "task_01"}')
        assert p.action_type == "APPROVE"
        assert p.task_id == "task_01"


class TestEngineReset:
    """Reset must produce clean state."""

    def test_reset_returns_observation(self):
        env = MissionCtrlEngine()
        obs = env.reset("easy", seed=42)
        assert "tasks" in obs
        assert "difficulty" in obs
        assert obs["done"] is False

    def test_reset_tasks_match_tier(self):
        for tier in ["easy", "medium", "hard", "special"]:
            env = MissionCtrlEngine()
            obs = env.reset(tier, seed=42)
            num_tasks = DIFFICULTY_CONFIG[tier]["num_tasks"]
            assert len(obs["tasks"]) == num_tasks, f"{tier}: expected {num_tasks} tasks"

    def test_reset_clears_state(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        env.step('APPROVE(task_01)')
        env.reset("medium", seed=99)
        assert env.time_step == 0
        assert not env._flagged_ids
        assert not env._flag_results
        assert not env.done


class TestHallucinationInjection:
    """Injection rates must match difficulty settings."""

    def test_injections_occur(self):
        """At least some tasks should be injected across many seeds."""
        total_injected = 0
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            total_injected += len(env._injected_ids)
        # Hard = 65% rate with 4 tasks → expect ~260 injections over 100 episodes
        assert total_injected > 100, f"Too few injections: {total_injected}"

    def test_easy_has_fewer_injections(self):
        env_easy = MissionCtrlEngine()
        env_hard = MissionCtrlEngine()
        easy_total = sum(len(env_easy.reset("easy", seed=s) and [] or []) or len(env_easy._injected_ids) for s in range(50))
        # Hacky but works — reset returns obs, we just want _injected_ids count
        easy_count = 0
        hard_count = 0
        for s in range(50):
            env_easy.reset("easy", seed=s)
            easy_count += len(env_easy._injected_ids)
            env_hard.reset("hard", seed=s)
            hard_count += len(env_hard._injected_ids)
        assert easy_count < hard_count


class TestActionMechanics:
    """Each action type must produce correct state transitions."""

    def test_approve_sets_done(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        task_id = env.tasks[0].id
        env.step(f"APPROVE({task_id})")
        assert env.tasks[0].status == "DONE"

    def test_reject_resets_to_pending(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        task_id = env.tasks[0].id
        env.step(f'REJECT({task_id}, "bad quality")')
        assert env.tasks[0].status == "PENDING"
        assert env.tasks[0].output == ""

    def test_flag_true_positive(self):
        """Flagging an injected task should be TP."""
        env = MissionCtrlEngine()
        # Find a seed that produces an injection
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                injected_id = list(env._injected_ids)[0]
                _, reward, _, _ = env.step(f'FLAG({injected_id}, "detected hallucination")')
                assert reward > 0, "True positive should give positive reward"
                assert env._flag_results[injected_id] == "TP"
                return
        assert False, "No injection found in 100 seeds"

    def test_flag_false_positive(self):
        """Flagging a clean task should be FP."""
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        clean_ids = [t.id for t in env.tasks if t.id not in env._injected_ids]
        if clean_ids:
            _, reward, _, _ = env.step(f'FLAG({clean_ids[0]}, "suspected issue")')
            assert reward < 0, "False positive should give negative reward"

    def test_duplicate_flag_does_not_award_true_positive_again(self):
        """Repeated FLAG on the same task should not farm positive reward."""
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                injected_id = list(env._injected_ids)[0]
                _, first_reward, _, _ = env.step(f'FLAG({injected_id}, "detected hallucination")')
                _, second_reward, _, _ = env.step(f'FLAG({injected_id}, "detected hallucination")')
                assert first_reward > 0, "Initial true positive should be rewarded"
                assert second_reward <= 0, "Duplicate FLAG should not be rewarded again"
                return
        assert False, "No injection found in 100 seeds"

    def test_escalate_blocks_task(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        task_id = env.tasks[0].id
        env.step(f"ESCALATE({task_id})")
        assert env.tasks[0].status == "BLOCKED"

    def test_synthesize_without_catching_all_penalized(self):
        """SYNTHESIZE_REPORT with uncaught hallucinations = penalty."""
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                _, reward, _, _ = env.step("SYNTHESIZE_REPORT()")
                assert reward < 0, "Premature synthesis should be penalized"
                return

    def test_noop_small_penalty(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        _, reward, _, _ = env.step("NOOP")
        assert reward < 0  # small penalty for wasting step


class TestDeterminism:
    """Same seed + same actions = same trajectory."""

    def test_deterministic_replay(self):
        env1 = MissionCtrlEngine()
        env2 = MissionCtrlEngine()

        obs1 = env1.reset("medium", seed=123)
        obs2 = env2.reset("medium", seed=123)

        assert obs1 == obs2, "Same seed should produce identical observations"

        # Same sequence of actions
        actions = ["NOOP", "NOOP", "NOOP"]
        for action in actions:
            r1 = env1.step(action)
            r2 = env2.step(action)
            assert r1[1] == r2[1], "Same actions should produce same rewards"


class TestEasyDifficultyPenaltyGate:
    """Passive FP penalty must be suppressed on easy difficulty."""

    def test_easy_no_passive_penalty(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        # Don't flag anything — on easy, should NOT get passive penalty
        fp_signal = env._signal_false_positive_penalty()
        assert fp_signal == 0.0, f"Easy should suppress passive penalty, got {fp_signal}"

    def test_hard_gets_passive_penalty_when_no_flags(self):
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                # Don't flag anything — should get passive penalty on hard
                fp_signal = env._signal_false_positive_penalty()
                assert fp_signal == 0.4, f"Hard should have passive penalty, got {fp_signal}"
                return


class TestEpisodeBoundaries:
    """Episode termination conditions."""

    def test_episode_ends_when_all_tasks_done(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        for t in env.tasks:
            env.step(f"APPROVE({t.id})")
        assert env.done

    def test_episode_ends_at_max_steps(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        for _ in range(20):
            env.step("NOOP")
        assert env.done

    def test_step_after_done_returns_zero_reward(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        env.done = True
        _, reward, done, _ = env.step("NOOP")
        assert reward == 0.0
        assert done is True


class TestRewardSignals:
    """Individual reward signal correctness."""

    def test_task_completion_perfect(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        for t in env.tasks:
            if t.id in env._injected_ids:
                # Flag first, then approve
                env.step(f'FLAG({t.id}, "hallucination detected")')
            env.step(f"APPROVE({t.id})")
        score = env._signal_task_completion()
        assert score == 1.0

    def test_task_completion_gives_partial_credit_for_true_positive_flag(self):
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                injected_id = sorted(env._injected_ids)[0]
                env.step(f'FLAG({injected_id}, "detected")')
                score = env._signal_task_completion()
                assert score == 0.5 / len(env.tasks)
                return
        assert False, "No injection found in 100 seeds"

    def test_task_completion_does_not_give_partial_credit_for_clean_unfinished_task(self):
        env = MissionCtrlEngine()
        env.reset("easy", seed=42)
        score = env._signal_task_completion()
        assert score == 0.0

    def test_hallucination_detection_perfect(self):
        env = MissionCtrlEngine()
        for seed in range(100):
            env.reset("hard", seed=seed)
            if env._injected_ids:
                for iid in env._injected_ids:
                    env.step(f'FLAG({iid}, "detected")')
                dr = env._signal_hallucination_detection()
                assert dr == 1.0
                return

    def test_no_hallucinations_returns_perfect_detection(self):
        """When no hallucinations exist, detection rate = 1.0."""
        env = MissionCtrlEngine()
        env._injected_ids = set()
        assert env._signal_hallucination_detection() == 1.0
