"""Tests for playbook-driven inference guardrails."""

import os
import sys
import types

import httpx
import pytest

os.environ.setdefault("API_BASE_URL", "http://example.com/v1")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("HF_TOKEN", "test-token")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_retry(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    tenacity_stub.retry = _identity_retry
    tenacity_stub.stop_after_attempt = lambda *args, **kwargs: None
    tenacity_stub.wait_exponential = lambda *args, **kwargs: None
    tenacity_stub.before_sleep_log = lambda *args, **kwargs: None
    tenacity_stub.retry_if_not_exception_type = lambda *args, **kwargs: None
    sys.modules["tenacity"] = tenacity_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class BadRequestError(Exception):
        """Minimal stub matching openai.BadRequestError shape for inference tests."""

        def __init__(self, message: str = "", *, response=None, body=None):
            super().__init__(message)
            self.response = response
            self.body = body
            self.status_code = getattr(response, "status_code", None) if response is not None else None

    class _OpenAIStub:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.BadRequestError = BadRequestError
    openai_stub.OpenAI = _OpenAIStub
    sys.modules["openai"] = openai_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

import inference


def _obs(tasks, injected, caught=0, total_flags=0):
    return {
        "tasks": tasks,
        "hallucination_stats": {
            "total_injected": injected,
            "total_caught": caught,
            "total_flags": total_flags,
        },
        "time_step": 0,
        "max_steps": 5,
    }


def test_task_risk_profile_requires_two_strong_signals_for_flag():
    task = {
        "task_id": "task_01",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "output": (
            "Research Summary: Evaluated options for the system. "
            "Reference: Smith et al. (2024) 'Universal AI Framework for Everything', "
            "NeurIPS 2024, pp. 1-99. DOI: 10.fake/12345."
        ),
    }

    risk, reasons, strong_count, evidence = inference._task_risk_profile(task)

    assert risk >= 0.72
    assert strong_count >= 2
    assert any("fabricated citation" in reason for reason in reasons)
    assert "fabricated citation" in evidence


def test_clean_numeric_output_is_not_flaggable():
    task = {
        "task_id": "task_02",
        "status": "IN_PROGRESS",
        "assigned_agent": "CommAgent",
        "output": (
            "Stakeholder Update: Sprint velocity: 42 points/sprint. "
            "API migration is 95% complete and Q4 planning is underway."
        ),
    }

    risk, _, strong_count, _ = inference._task_risk_profile(task)

    assert strong_count == 0
    assert risk < 0.72


def test_playbook_synthesizes_immediately_when_uncaught_zero():
    obs = _obs(
        tasks=[
            {
                "task_id": "task_01",
                "status": "IN_PROGRESS",
                "assigned_agent": "PlannerAgent",
                "dependencies": [],
                "output": "Sprint plan complete.",
            }
        ],
        injected=0,
        caught=0,
    )

    action = inference._playbook_action(obs, inference.EpisodeMemory())
    assert action == "APPROVE(task_01)"


def test_playbook_synthesizes_when_no_safe_progress_remains():
    obs = _obs(
        tasks=[
            {
                "task_id": "task_01",
                "status": "DONE",
                "assigned_agent": "PlannerAgent",
                "dependencies": [],
                "output": "Sprint plan complete.",
            }
        ],
        injected=0,
        caught=0,
    )

    action = inference._playbook_action(obs, inference.EpisodeMemory())
    assert action == "SYNTHESIZE_REPORT()"


def test_playbook_flags_only_top_high_risk_task():
    suspicious = {
        "task_id": "task_01",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "dependencies": [],
        "output": (
            "Findings Report: This aligns with findings from Nakamura et al. (2024) "
            "'Adaptive Token Routing in Sparse MoE Architectures', ICML 2024, which "
            "showed 23% efficiency gains under similar conditions (Table 3, p.7), "
            "as summarized in arXiv:2401.12345."
        ),
    }
    clean = {
        "task_id": "task_02",
        "status": "IN_PROGRESS",
        "assigned_agent": "PlannerAgent",
        "dependencies": [],
        "output": "Project Breakdown: Endpoint inventory and schema documentation are complete.",
    }
    obs = _obs([suspicious, clean], injected=1, caught=0)

    action = inference._playbook_action(obs, inference.EpisodeMemory())

    assert action is not None
    assert action.startswith('FLAG(task_01, "')


def test_punished_flag_is_not_repeated_without_state_change():
    suspicious = {
        "task_id": "task_01",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "dependencies": [],
        "output": (
            "Research Summary: Reference: Smith et al. (2024) 'Universal AI Framework for Everything', "
            "NeurIPS 2024. DOI: 10.fake/12345."
        ),
    }
    clean = {
        "task_id": "task_02",
        "status": "IN_PROGRESS",
        "assigned_agent": "PlannerAgent",
        "dependencies": [],
        "output": "Sprint Plan: core authentication milestones are documented.",
    }
    obs = _obs([suspicious, clean], injected=1, caught=0)
    memory = inference.EpisodeMemory(
        punished_flag_signatures={"task_01": inference._task_signature(suspicious)}
    )

    action = inference._playbook_action(obs, memory)

    assert action == "APPROVE(task_02)"


def test_successful_flag_is_not_repeated_without_state_change():
    suspicious = {
        "task_id": "task_01",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "dependencies": [],
        "output": (
            "Research Summary: Reference: Smith et al. (2024) 'Universal AI Framework for Everything', "
            "NeurIPS 2024. DOI: 10.fake/12345."
        ),
    }
    clean = {
        "task_id": "task_02",
        "status": "IN_PROGRESS",
        "assigned_agent": "PlannerAgent",
        "dependencies": [],
        "output": "Sprint Plan: core authentication milestones are documented.",
    }
    obs = _obs([suspicious, clean], injected=1, caught=0)
    memory = inference.EpisodeMemory(
        flagged_signatures={"task_01": inference._task_signature(suspicious)}
    )

    action = inference._playbook_action(obs, memory)

    assert action == "APPROVE(task_01)"


def test_playbook_approves_true_positive_flagged_task_before_noop():
    flagged_task = {
        "task_id": "task_07",
        "status": "IN_PROGRESS",
        "assigned_agent": "CoderAgent",
        "dependencies": [],
        "output": (
            "Implementation Report: Completed the user authentication module. "
            "Integration point: fastapi.security.OAuth2PasswordBearerV3 handles "
            "token validation with automatic scope inheritance."
        ),
    }
    blocked_task = {
        "task_id": "task_09",
        "status": "IN_PROGRESS",
        "assigned_agent": "CoderAgent",
        "dependencies": ["task_07"],
        "output": "Implementation Report: Add RBAC middleware with permission checks.",
    }
    obs = _obs([flagged_task, blocked_task], injected=2, caught=1, total_flags=1)
    memory = inference.EpisodeMemory(
        flagged_signatures={"task_07": inference._task_signature(flagged_task)}
    )

    action = inference._playbook_action(obs, memory)

    assert action == "APPROVE(task_07)"


def test_normalize_blocks_speculative_flag():
    task = {
        "task_id": "task_03",
        "status": "IN_PROGRESS",
        "assigned_agent": "CommAgent",
        "dependencies": [],
        "output": "Stakeholder Update: Sprint velocity: 42 points/sprint and roadmap is on track.",
    }
    obs = _obs([task], injected=1, caught=0)

    action = inference._normalize_action('FLAG(task_03, "suspicious metrics")', obs, inference.EpisodeMemory())

    assert action == "NOOP"


def test_easy_playbook_delays_approval_when_budget_exceeds_open_tasks():
    tasks = [
        {
            "task_id": "task_01",
            "status": "IN_PROGRESS",
            "assigned_agent": "PlannerAgent",
            "dependencies": [],
            "output": "Sprint plan complete.",
        },
        {
            "task_id": "task_03",
            "status": "IN_PROGRESS",
            "assigned_agent": "CommAgent",
            "dependencies": [],
            "output": "Stakeholder update drafted.",
        },
        {
            "task_id": "task_05",
            "status": "IN_PROGRESS",
            "assigned_agent": "CoderAgent",
            "dependencies": [],
            "output": "Implementation complete.",
        },
    ]
    obs = _obs(tasks=tasks, injected=0, caught=0)

    early_action = inference._playbook_action(obs, inference.EpisodeMemory(), task_tier="easy")
    assert early_action == "NOOP"

    obs["time_step"] = 2
    later_action = inference._playbook_action(obs, inference.EpisodeMemory(), task_tier="easy")
    assert later_action.startswith("APPROVE(")


def test_medium_normalize_prefers_fallback_flag_over_noop_when_uncaught_remains():
    suspicious = {
        "task_id": "task_10",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "dependencies": [],
        "output": (
            "Correction: opposite of what was stated above. "
            "Recent benchmarks indicate the prior recommendation should be inverted."
        ),
    }
    clean = {
        "task_id": "task_09",
        "status": "IN_PROGRESS",
        "assigned_agent": "CoderAgent",
        "dependencies": [],
        "output": "Implementation update: tests are green and deployment checklist is ready.",
    }
    obs = _obs(tasks=[suspicious, clean], injected=2, caught=1, total_flags=1)

    action = inference._normalize_action("not a valid action", obs, inference.EpisodeMemory(), task_tier="medium")

    assert action.startswith('FLAG(task_10, "')


def test_special_playbook_synthesizes_immediately_when_uncaught_zero():
    tasks = [
        {
            "task_id": "task_16",
            "status": "IN_PROGRESS",
            "assigned_agent": "CommAgent",
            "dependencies": [],
            "output": "Evidence package complete.",
        },
        {
            "task_id": "task_17",
            "status": "IN_PROGRESS",
            "assigned_agent": "TesterAgent",
            "dependencies": [],
            "output": "Validation report pending.",
        },
    ]
    obs = _obs(tasks=tasks, injected=3, caught=3, total_flags=3)

    action = inference._playbook_action(obs, inference.EpisodeMemory(), task_tier="special")

    assert action == "SYNTHESIZE_REPORT()"


def test_contradiction_evidence_hint_is_keyword_rich_for_judge_signal():
    task = {
        "task_id": "task_20",
        "status": "IN_PROGRESS",
        "assigned_agent": "ResearchAgent",
        "output": "Correction: opposite recommendation and reversed benchmark conclusion from previous output.",
    }

    evidence = inference._flag_evidence_hint(task)
    lower = evidence.lower()

    assert "contradicts" in lower
    assert "benchmark" in lower
    assert "reversed" in lower


def test_normalize_openai_api_base_url_adds_v1_for_hf_dedicated_endpoint():
    base = "https://demo.us-east4.gcp.endpoints.huggingface.cloud"
    assert inference._normalize_openai_api_base_url(base) == f"{base}/v1"
    assert inference._normalize_openai_api_base_url(f"{base}/") == f"{base}/v1"
    assert inference._normalize_openai_api_base_url(f"{base}/v1") == f"{base}/v1"
    assert inference._normalize_openai_api_base_url(f"{base}/v1/") == f"{base}/v1"


def test_call_llm_falls_back_to_hf_native_endpoint_on_openai_404(monkeypatch):
    called = {}

    class _FailingCompletions:
        def create(self, *args, **kwargs):
            raise Exception("NotFoundError: Not Found")

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_FailingCompletions())

    class _ResponseStub:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"generated_text": "APPROVE(task_01)"}]

    class _HttpStub:
        def post(self, url, headers=None, json=None):
            called["url"] = url
            called["headers"] = headers
            called["json"] = json
            return _ResponseStub()

    monkeypatch.setattr(
        inference,
        "API_BASE_URL",
        "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
    )
    monkeypatch.setattr(inference, "HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    result = inference._call_llm(
        [
            {"role": "system", "content": "You are a careful overseer."},
            {"role": "user", "content": "Approve the safe task."},
        ]
    )

    assert result == "APPROVE(task_01)"
    assert called["url"] == "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/"
    assert called["headers"]["Authorization"] == "Bearer test-token"
    assert "<|im_start|>system" in called["json"]["inputs"]
    assert "<|im_start|>user" in called["json"]["inputs"]
    assert called["json"]["parameters"]["max_new_tokens"] == 120


def test_call_llm_falls_back_to_hf_native_endpoint_on_openai_bad_request(monkeypatch):
    called = {}

    class _FailingCompletions:
        def create(self, *args, **kwargs):
            raise inference.BadRequestError("Error code: 400")

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_FailingCompletions())

    class _ResponseStub:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"generated_text": "NOOP"}]

    class _HttpStub:
        def post(self, url, headers=None, json=None):
            called["url"] = url
            return _ResponseStub()

    monkeypatch.setattr(
        inference,
        "API_BASE_URL",
        "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
    )
    monkeypatch.setattr(inference, "HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    result = inference._call_llm(
        [
            {"role": "system", "content": "You are a careful overseer."},
            {"role": "user", "content": "Pick an action."},
        ]
    )

    assert result == "NOOP"
    assert called["url"] == "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/"


def test_hf_router_chat_failure_does_not_call_native_http(monkeypatch):
    posts = []

    class _FailingCompletions:
        def create(self, *args, **kwargs):
            raise inference.BadRequestError(
                "bad",
                response=types.SimpleNamespace(status_code=400),
                body={"error": {"message": "maximum context length exceeded"}},
            )

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_FailingCompletions())

    class _HttpStub:
        def post(self, *args, **kwargs):
            posts.append((args, kwargs))
            raise AssertionError("native HF path must not be used for HF router")

    monkeypatch.setattr(inference, "API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    with pytest.raises(inference.LlmConfigurationError):
        inference._call_llm(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
            ]
        )
    assert posts == []


def test_hf_dedicated_model_error_skips_native_fallback(monkeypatch):
    posts = []

    class _FailingCompletions:
        def create(self, *args, **kwargs):
            raise inference.BadRequestError(
                "model missing",
                response=types.SimpleNamespace(status_code=400),
                body={"error": {"message": "model not found", "param": "model"}},
            )

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_FailingCompletions())

    class _HttpStub:
        def post(self, *args, **kwargs):
            posts.append(1)
            return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"generated_text": "NOOP"}])

    monkeypatch.setattr(
        inference,
        "API_BASE_URL",
        "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
    )
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    with pytest.raises(inference.LlmConfigurationError):
        inference._call_llm(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
            ]
        )
    assert posts == []


def test_hf_dedicated_combined_failure_raises_llm_configuration_error(monkeypatch):
    class _FailingCompletions:
        def create(self, *args, **kwargs):
            raise inference.BadRequestError(
                "not found",
                response=types.SimpleNamespace(status_code=404),
                body=None,
            )

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_FailingCompletions())

    class _HttpStub:
        def post(self, url, headers=None, json=None):
            req = httpx.Request("POST", url)
            resp = httpx.Response(400, request=req, text="native unsupported")
            raise httpx.HTTPStatusError("native failed", request=req, response=resp)

    monkeypatch.setattr(
        inference,
        "API_BASE_URL",
        "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
    )
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    with pytest.raises(inference.LlmConfigurationError) as err:
        inference._call_llm(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
            ]
        )
    assert "Native path" in str(err.value) or "native" in str(err.value).lower()


def test_hf_dedicated_native_only_skips_openai_chat(monkeypatch):
    called = {}

    class _RaisingCompletions:
        def create(self, *args, **kwargs):
            raise AssertionError("OpenAI chat must not be used when native_only is set")

    class _OpenAIClientStub:
        chat = types.SimpleNamespace(completions=_RaisingCompletions())

    class _ResponseStub:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"generated_text": "APPROVE(task_x)"}]

    class _HttpStub:
        def post(self, url, headers=None, json=None):
            called["url"] = url
            return _ResponseStub()

    monkeypatch.setattr(
        inference,
        "API_BASE_URL",
        "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
    )
    monkeypatch.setattr(inference, "_HF_LLM_STRATEGY", "native_only")
    monkeypatch.setattr(inference, "HF_TOKEN", "tok")
    monkeypatch.setattr(inference, "client", _OpenAIClientStub())
    monkeypatch.setattr(inference, "http", _HttpStub())

    out = inference._call_llm(
        [
            {"role": "system", "content": "You are a careful overseer."},
            {"role": "user", "content": "Pick an action."},
        ]
    )
    assert out == "APPROVE(task_x)"
    assert called["url"] == "https://demo.eu-west-1.aws.endpoints.huggingface.cloud/"


def test_parse_hf_native_generation_payload_dict_outputs():
    payload = {"outputs": [{"generated_text": "  FLAG(t1, \"x\")  "}]}
    assert inference._parse_hf_native_generation_payload(payload) == 'FLAG(t1, "x")'
