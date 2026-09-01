"""
Tests for 5.2 Structured Outputs & Tools.
"""

import pytest
from layer_1.phase5.common.types import Message, ToolCall
from layer_1.phase5.common.llm_backend import get_llm_backend
from layer_1.phase5.structured_outputs_and_tools.structured_outputs import (
    JSONExtractor,
    StructuredOutputEngine,
    FinancialRiskAssessment,
    UserProfile,
)
from layer_1.phase5.structured_outputs_and_tools.tool_schemas_and_dispatch import (
    ToolRegistry,
    ToolExecutor,
    global_registry,
    generate_tool_schema,
)
from layer_1.phase5.structured_outputs_and_tools.tool_loop_engine import AgentToolLoopEngine
from layer_1.phase5.structured_outputs_and_tools.tool_simulator_and_failures import ToolFailureSimulator


def test_json_extractor_and_repair():
    raw_markdown = "Here is the result:\n```json\n{\"risk_level\": \"LOW\", \"risk_score\": 12.0}\n```\nHope that helps!"
    extracted = JSONExtractor.extract_json_string(raw_markdown)
    assert extracted == '{"risk_level": "LOW", "risk_score": 12.0}'

    broken_trailing_comma = '{"a": 1, "b": 2,}'
    repaired = JSONExtractor.attempt_syntactic_repair(broken_trailing_comma)
    assert repaired == '{"a": 1, "b": 2}'

    unclosed_brace = '{"a": 1, "b": 2'
    repaired_brace = JSONExtractor.attempt_syntactic_repair(unclosed_brace)
    assert repaired_brace == '{"a": 1, "b": 2}'


def test_structured_output_engine():
    backend = get_llm_backend("mock")
    engine = StructuredOutputEngine(backend=backend, max_retries=3)

    prompt = [
        Message.system("Evaluate user profile."),
        Message.user("Get profile for user alice_dev")
    ]
    res = engine.parse_structured(UserProfile, prompt)
    assert res.success is True
    assert res.parsed_object is not None
    assert res.parsed_object.username == "alice_dev"
    assert res.parsed_object.credit_score == 760


def test_tool_reflection_and_rbac():
    reg = ToolRegistry()

    @reg.register(name="custom_math", description="Multiplies numbers", required_permissions=["math:exec"])
    def multiply(a: int, b: int) -> int:
        return a * b

    schema = reg.get("custom_math").definition.parameters
    assert "properties" in schema
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]

    executor = ToolExecutor(reg)

    # Missing permission
    tc = ToolCall(name="custom_math", arguments={"a": 3, "b": 4})
    res_blocked = executor.execute_tool_call(tc, user_permissions=set())
    assert res_blocked.is_blocked_by_auth is True
    assert res_blocked.success is False

    # Authorized
    res_ok = executor.execute_tool_call(tc, user_permissions={"math:exec"})
    assert res_ok.success is True
    assert res_ok.output == 12


def test_agent_tool_loop():
    backend = get_llm_backend("mock")
    engine = AgentToolLoopEngine(backend=backend, registry=global_registry, max_steps=5)
    summary = engine.run(user_query="Calculate volatility for metric")
    assert summary.success is True
    assert summary.tool_calls_executed >= 1
    assert "volatility" in summary.final_answer.lower() or "tool execution" in summary.final_answer.lower()


def test_tool_failure_simulator():
    sim = ToolFailureSimulator()
    results = sim.run_all_tests()
    assert len(results) == 5
    for r in results:
        assert r.handled_safely is True
