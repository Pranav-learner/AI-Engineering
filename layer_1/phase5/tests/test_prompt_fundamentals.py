"""
Tests for 5.1 Prompt Fundamentals.
"""

import pytest
from layer_1.phase5.common.types import Message, MessageRole
from layer_1.phase5.common.llm_backend import get_llm_backend
from layer_1.phase5.prompt_fundamentals.prompt_structure import (
    StructuredPrompt,
    OutputConstraint,
    PromptFormat,
)
from layer_1.phase5.prompt_fundamentals.prompt_strategies import (
    ZeroShotStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
    SelfConsistencyStrategy,
    TreeOfThoughtStrategy,
    PlanAndSolveStrategy,
)
from layer_1.phase5.prompt_fundamentals.prompt_experiments_and_failures import PromptFailureHarness


def test_structured_prompt_compilation():
    prompt = StructuredPrompt(system_instruction="You are a helpful analyst.")
    prompt.add_context("Account Data", "Balance is $500.")
    prompt.add_exemplar("Sample task", "Sample answer")
    prompt.set_user_instruction("Evaluate transfer.")
    prompt.set_output_constraint(OutputConstraint(format_type="json", max_length_words=100))

    messages = prompt.compile_messages()
    assert len(messages) >= 4
    assert messages[0].role == MessageRole.SYSTEM
    assert "Output Requirements" in messages[0].content
    assert messages[1].role == MessageRole.USER
    assert messages[2].role == MessageRole.ASSISTANT
    assert "<context>" in messages[-1].content
    assert "<task>" in messages[-1].content

    xml_text = prompt.compile_text(PromptFormat.XML_WRAPPED)
    assert "<system>" in xml_text
    assert "<user_query>" in xml_text


def test_prompt_strategies():
    backend = get_llm_backend("mock")

    # Zero-shot
    zs = ZeroShotStrategy.build_prompt("Solve task")
    assert len(zs) == 2

    # Few-shot
    fs = FewShotStrategy()
    fs.add_exemplar("ex_in", "ex_out")
    fs_msgs = fs.build_prompt("task")
    assert len(fs_msgs) == 4

    # CoT
    cot_msgs = ChainOfThoughtStrategy.build_zero_shot_cot("task")
    assert "think step by step" in cot_msgs[-1].content

    # Self-Consistency
    sc = SelfConsistencyStrategy(backend, num_samples=3)
    sc_res = sc.execute("Compute math")
    assert sc_res.total_samples == 3
    assert sc_res.winning_answer != ""
    assert sc_res.confidence > 0.0

    # Tree-of-Thought
    tot = TreeOfThoughtStrategy(backend, branch_factor=2, max_depth=2)
    leaf, path = tot.search("Hedging strategy")
    assert len(path) == 3
    assert leaf.depth == 2

    # Plan-and-Solve
    planner = PlanAndSolveStrategy(backend)
    plan_out = planner.execute("Audit account")
    assert len(plan_out["steps"]) == 3
    assert "Plan Execution Completed" in plan_out["synthesis"]


def test_prompt_failure_modes_harness():
    harness = PromptFailureHarness()
    results = harness.run_all_experiments()
    assert len(results) == 5
    for r in results:
        assert r.passed is True
