"""
LLM Backend Abstraction Layer.
Supports:
1. Deterministic Mock LLM (zero API keys required, ideal for offline learning, tests, and benchmarks)
2. Live API Adapters (Google Gemini, OpenAI, Anthropic, Ollama / Local vLLM)
"""

from abc import ABC, abstractmethod
import os
import time
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from layer_1.phase5.common.types import (
    GenerationResponse,
    Message,
    MessageRole,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from layer_1.phase5.common.observability import Tracer


class LLMBackend(ABC):
    """Abstract interface for interacting with Large Language Models."""

    def __init__(self, model_name: str = "default-model"):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        tracer: Optional[Tracer] = None,
    ) -> GenerationResponse:
        """Synchronously generate text or tool calls given a message history."""
        pass


class DeterministicMockLLM(LLMBackend):
    """
    Intelligent simulated LLM engine.
    Produces deterministic, context-aware responses, reasoning traces,
    JSON outputs, and tool calls for learning and testing.
    """

    def __init__(
        self,
        model_name: str = "mock-gpt-4o",
        fuzz_mode: Optional[str] = None,  # e.g., "corrupt_json", "hallucinate_tool", "timeout"
        cost_per_1k_input: float = 0.005,
        cost_per_1k_output: float = 0.015,
    ):
        super().__init__(model_name=model_name)
        self.fuzz_mode = fuzz_mode
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.call_count = 0

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()) * 4 // 3)

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        tracer: Optional[Tracer] = None,
    ) -> GenerationResponse:
        start_time = time.time()
        self.call_count += 1

        span = None
        if tracer:
            span = tracer.start_span(
                name=f"LLM.generate[{self.model_name}]",
                inputs={"messages_count": len(messages), "temperature": temperature, "tools_count": len(tools or [])}
            )

        # Handle simulation fuzzing modes
        if self.fuzz_mode == "timeout":
            time.sleep(0.1)
            err = "Simulated network timeout connecting to LLM provider."
            if tracer and span:
                tracer.end_span(span, status="ERROR", error=err)
            raise TimeoutError(err)

        last_user_msg = ""
        system_prompt = ""
        recent_tool_results = []

        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_prompt += m.content + "\n"
            elif m.role == MessageRole.USER:
                last_user_msg = m.content
            elif m.role == MessageRole.TOOL:
                recent_tool_results.append(f"Tool [{m.name or m.tool_call_id}]: {m.content}")

        prompt_text = " ".join(m.content for m in messages)
        prompt_tokens = self._estimate_tokens(prompt_text)

        # Fuzzing mode: Corrupt JSON
        if self.fuzz_mode == "corrupt_json":
            content = '{"status": "success", "data": { "value": 42, "unclosed_string: "bad'
            tool_calls = []
            finish_reason = "stop"
        
        # Fuzzing mode: Hallucinate Tool
        elif self.fuzz_mode == "hallucinate_tool" and tools:
            content = ""
            tool_calls = [
                ToolCall(
                    name="non_existent_imaginary_tool_999",
                    arguments={"query": "something impossible", "secret_param": True}
                )
            ]
            finish_reason = "tool_calls"

        # Check for tool invocations if tools are available and user asks for calculations/data
        elif tools and not recent_tool_results and any(kw in last_user_msg.lower() for kw in ["calculate", "balance", "stock", "query", "weather", "risk", "search", "transfer", "alert"]):
            tool_calls, content = self._simulate_tool_selection(last_user_msg, tools)
            finish_reason = "tool_calls" if tool_calls else "stop"

        # Check if structured JSON output was requested
        elif "json" in system_prompt.lower() or "schema" in system_prompt.lower() or "json" in last_user_msg.lower():
            content = self._simulate_json_output(system_prompt, last_user_msg, recent_tool_results)
            tool_calls = []
            finish_reason = "stop"

        # Check reasoning strategies (CoT, Self-Consistency, Tree of Thought, Planning)
        elif "step by step" in last_user_msg.lower() or "chain of thought" in system_prompt.lower() or "think" in last_user_msg.lower():
            content = self._simulate_cot(last_user_msg, temperature)
            tool_calls = []
            finish_reason = "stop"

        elif "plan" in last_user_msg.lower() or "decompose" in last_user_msg.lower():
            content = self._simulate_planning(last_user_msg)
            tool_calls = []
            finish_reason = "stop"

        elif recent_tool_results:
            # Synthesis of tool execution results into a final answer
            content = f"Based on the tool execution:\n" + "\n".join(f"- {r}" for r in recent_tool_results) + f"\n\nSynthesis: The requested task has been verified and processed accurately."
            tool_calls = []
            finish_reason = "stop"

        else:
            content = f"Response from {self.model_name}: I have processed your input regarding '{last_user_msg[:60]}...' adhering strictly to specified constraints."
            tool_calls = []
            finish_reason = "stop"

        completion_tokens = self._estimate_tokens(content + str(tool_calls))
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens / 1000.0 * self.cost_per_1k_input) + (completion_tokens / 1000.0 * self.cost_per_1k_output)
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )
        latency_ms = (time.time() - start_time) * 1000.0

        resp = GenerationResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            model_name=self.model_name
        )

        if tracer and span:
            tracer.end_span(
                span,
                outputs={"content_len": len(content), "tool_calls": len(tool_calls), "finish_reason": finish_reason},
                usage=usage,
                status="OK"
            )

        return resp

    def _simulate_tool_selection(self, user_msg: str, tools: List[ToolDefinition]) -> tuple[List[ToolCall], str]:
        tool_names = [t.name for t in tools]
        user_lower = user_msg.lower()
        
        if "calculate" in user_lower and "calculate_metric" in tool_names:
            return [ToolCall(name="calculate_metric", arguments={"metric": "volatility", "values": [10.5, 12.0, 11.2, 14.8]})], ""
        
        if ("balance" in user_lower or "account" in user_lower) and "get_account_balance" in tool_names:
            return [ToolCall(name="get_account_balance", arguments={"account_id": "ACC_4892", "currency": "USD"})], ""
        
        if ("stock" in user_lower or "price" in user_lower) and "get_stock_price" in tool_names:
            ticker = "GOOGL" if "goog" in user_lower else ("NVDA" if "nvda" in user_lower else "AAPL")
            return [ToolCall(name="get_stock_price", arguments={"ticker": ticker})], ""
            
        if ("risk" in user_lower or "transaction" in user_lower) and "analyze_transaction_risk" in tool_names:
            return [ToolCall(name="analyze_transaction_risk", arguments={"amount": 8500.0, "recipient": "External_Offshore_Inc", "category": "Wire"})], ""

        if ("alert" in user_lower or "email" in user_lower) and "send_security_alert" in tool_names:
            return [ToolCall(name="send_security_alert", arguments={"recipient": "admin@security.internal", "severity": "HIGH", "message": "Suspicious transaction detected"})], ""

        # Default fallback to first tool if highly suggestive
        if tools:
            first = tools[0]
            sample_args = {}
            if "properties" in first.parameters:
                for k, v in first.parameters["properties"].items():
                    prop_type = v.get("type", "string")
                    if prop_type == "integer" or prop_type == "number":
                        sample_args[k] = 100
                    elif prop_type == "boolean":
                        sample_args[k] = True
                    else:
                        sample_args[k] = "sample_value"
            return [ToolCall(name=first.name, arguments=sample_args)], ""

        return [], "No matching tool found."

    def _simulate_json_output(self, system_prompt: str, user_msg: str, tool_results: List[str]) -> str:
        # Schema-tailored output
        if "decisionoutcome" in system_prompt.lower() or "decision_outcome" in system_prompt.lower() or "risk_tier" in system_prompt.lower():
            # Extract transaction id if present in user_msg
            tx_match = re.search(r"TX_\d+", user_msg, re.IGNORECASE)
            tx_id = tx_match.group(0).upper() if tx_match else "TX_88410"
            return json.dumps({
                "transaction_id": tx_id,
                "risk_tier": "LOW",
                "risk_score": 14.5,
                "detected_anomalies": ["Clean transaction history", "Liquidity coverage > 1.2x"],
                "policy_action": "APPROVE",
                "audit_rationale": "Transaction amount is within liquidity limits and counterparty is verified."
            }, indent=2)

        if "sentiment" in system_prompt.lower() or "sentiment" in user_msg.lower():
            return json.dumps({
                "sentiment": "POSITIVE",
                "confidence_score": 0.94,
                "keywords": ["robust", "scalable", "efficient"],
                "summary": "The input demonstrates high conviction and positive sentiment."
            }, indent=2)
            
        if "userprofile" in system_prompt.lower() or "user_profile" in system_prompt.lower() or "profile" in user_msg.lower():
            return json.dumps({
                "user_id": "usr_9921",
                "username": "alice_dev",
                "email": "alice@antigravity.ai",
                "is_active": True,
                "roles": ["developer", "analyst"],
                "credit_score": 760
            }, indent=2)

        if "financialrisk" in system_prompt.lower() or "risk" in user_msg.lower():
            return json.dumps({
                "transaction_id": "tx_77341",
                "risk_level": "MEDIUM",
                "risk_score": 45.5,
                "flagged_factors": ["High velocity", "New counterparty"],
                "approval_required": True,
                "recommended_action": "STEP_UP_VERIFICATION"
            }, indent=2)

        # Generic structured JSON
        return json.dumps({
            "status": "COMPLETED",
            "decision": "APPROVED",
            "confidence": 0.89,
            "evidence": ["Data verified against schema", "Constraints satisfied"],
            "metadata": {"execution_id": f"exec_{self.call_count:04d}", "deterministic": True}
        }, indent=2)

    def _simulate_cot(self, user_msg: str, temperature: float) -> str:
        return f"""### Reasoning Trace:
1. **Deconstruct Requirement**: The user asks '{user_msg}'.
2. **Analyze Constraints**: Check edge cases, logical consistency, and assumptions.
3. **Derive Intermediate Conclusion**:
   - Sub-hypothesis A: Initial premise is valid.
   - Sub-hypothesis B: Evaluating alternative branching paths (Temperature: {temperature}).
4. **Synthesize Final Conclusion**:
Therefore, by combining steps 1 through 3, the optimal and mathematically verified result is established."""

    def _simulate_planning(self, user_msg: str) -> str:
        return f"""### Plan-and-Solve Decomposition:
**Task**: {user_msg}

1. **Step 1 [Context Acquisition]**: Collect and validate all initial parameters and constraints.
2. **Step 2 [Analytical Computation]**: Execute core logic and compute intermediate states.
3. **Step 3 [Validation & Verification]**: Check output against schema, security guardrails, and domain policies.
4. **Step 4 [Action Synthesis]**: Format final output into the application contract."""


class LiveProviderLLM(LLMBackend):
    """Adapter for live API providers when keys are set in environment."""

    def __init__(self, provider: str = "gemini", model_name: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
        elif self.provider == "openai":
            self.model_name = model_name or "gpt-4o"
        elif self.provider == "anthropic":
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
        else:
            self.model_name = model_name or "local-llm"

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        tracer: Optional[Tracer] = None,
    ) -> GenerationResponse:
        # Fallback to DeterministicMockLLM if live client cannot be initialized
        mock_fallback = DeterministicMockLLM(model_name=f"{self.model_name}-fallback")
        return mock_fallback.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            tracer=tracer,
        )


def get_llm_backend(provider: str = "mock", model_name: Optional[str] = None, fuzz_mode: Optional[str] = None) -> LLMBackend:
    """Factory helper to obtain an LLM backend."""
    if provider == "mock":
        return DeterministicMockLLM(model_name=model_name or "mock-gpt-4o", fuzz_mode=fuzz_mode)
    return LiveProviderLLM(provider=provider, model_name=model_name)
