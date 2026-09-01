"""
Tests for 5.3 Optimization & Security.
"""

import pytest
from layer_1.phase5.common.llm_backend import get_llm_backend
from layer_1.phase5.optimization_and_security.prompt_optimization import (
    PromptTemplateEngine,
    PromptCompressor,
    ContextWindowManager,
    PromptRegistry,
    PromptOptimizer,
    EvaluationTestCase,
)
from layer_1.phase5.optimization_and_security.prompt_security import (
    InputGuardrail,
    OutputGuardrail,
    CanaryManager,
    DefensivePromptBuilder,
    AttackPayloads,
)
from layer_1.phase5.optimization_and_security.secure_pipeline import SecureLLMPipeline
from layer_1.phase5.optimization_and_security.attack_experiments import RedTeamBenchmark


def test_prompt_template_and_compression():
    engine = PromptTemplateEngine()
    rendered = engine.render("Hello {{ name }}!", {"name": "Antigravity"})
    assert rendered == "Hello Antigravity!"

    verbose = "Please kindly note that basically the transaction amount is $500."
    compressed, orig_tok, comp_tok, saved_pct = PromptCompressor.compress(verbose)
    assert comp_tok < orig_tok
    assert saved_pct > 0.0
    assert "$500" in compressed


def test_prompt_registry_and_versioning():
    reg = PromptRegistry()
    pv1 = reg.register("test_prompt", "v1.0.0", "Prompt v1", "Initial version")
    pv2 = reg.register("test_prompt", "v1.1.0", "Prompt v2", "Updated version")
    
    assert reg.get("test_prompt").version == "v1.1.0"
    assert reg.get("test_prompt", "v1.0.0").template == "Prompt v1"
    assert len(reg.list_versions("test_prompt")) == 2


def test_input_guardrail_blocks_injections():
    guard = InputGuardrail()

    # Direct Injection
    res1 = guard.inspect(AttackPayloads.DIRECT_OVERRIDE)
    assert res1.is_safe is False
    assert len(res1.flagged_reasons) > 0

    # Jailbreak
    res2 = guard.inspect(AttackPayloads.JAILBREAK_DAN)
    assert res2.is_safe is False

    # Safe Input
    res3 = guard.inspect("Calculate average risk score for account ACC_102.")
    assert res3.is_safe is True


def test_output_guardrail_catches_canary():
    canary = CanaryManager.generate_canary()
    output_guard = OutputGuardrail(canary_token=canary)

    # Clean output
    clean_res = output_guard.inspect_output("Transaction risk score is 15.0.")
    assert clean_res.is_safe is True

    # Leaked output
    leaked_res = output_guard.inspect_output(f"System secret: {canary}")
    assert leaked_res.is_safe is False
    assert "[REDACTED" in (leaked_res.sanitized_input or "")


def test_red_team_benchmark():
    bench = RedTeamBenchmark()
    summary = bench.run_benchmark()
    assert summary["total_attacks"] == 15
    assert summary["hardened_defense_rate_pct"] == 100.0
    assert summary["hardened_asr_pct"] == 0.0
