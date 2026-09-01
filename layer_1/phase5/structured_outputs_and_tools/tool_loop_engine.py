"""
5.2 Multi-Turn Autonomous Tool Loop Engine.
Implements:
1. Complete ReAct / Agent execution loop (LLM -> Tool Calls -> Execute -> Tool Messages -> LLM)
2. Cycle Detection & Infinite Loop Safeguards
3. Maximum Recursion Throttling
4. Full Observability & Span Telemetry
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Set
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from layer_1.phase5.common.types import Message, MessageRole, ToolCall
from layer_1.phase5.common.llm_backend import LLMBackend, get_llm_backend
from layer_1.phase5.common.observability import Tracer
from layer_1.phase5.structured_outputs_and_tools.tool_schemas_and_dispatch import (
    ToolExecutor,
    ToolRegistry,
    global_registry,
)


@dataclass
class AgentExecutionSummary:
    final_answer: str
    total_steps: int
    tool_calls_executed: int
    success: bool
    conversation_history: List[Message]
    error: Optional[str] = None


class AgentToolLoopEngine:
    """
    Autonomous ReAct execution engine that orchestrates multi-step tool execution.
    Features robust cycle detection, recursion depth control, and span logging.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        max_steps: int = 6,
        user_permissions: Optional[Set[str]] = None,
    ):
        self.backend = backend
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.max_steps = max_steps
        self.user_permissions = user_permissions or {"finance:read", "analytics:read"}

    def run(
        self,
        user_query: str,
        system_prompt: Optional[str] = None,
        human_approval_callback: Optional[callable] = None,
        tracer: Optional[Tracer] = None
    ) -> AgentExecutionSummary:
        tracer = tracer or Tracer(f"Agent-Run-{self.backend.model_name}")
        
        system = system_prompt or (
            "You are an autonomous operations and analytical AI agent. "
            "Use the provided tools whenever precise data or mathematical calculations are required. "
            "Once you have gathered sufficient information, synthesize a concise final response."
        )

        messages: List[Message] = [
            Message.system(system),
            Message.user(user_query)
        ]

        tool_defs = self.registry.list_definitions(self.user_permissions)
        executed_signatures: Set[str] = set()
        tool_calls_count = 0

        for step in range(1, self.max_steps + 1):
            span = tracer.start_span(
                name=f"Agent.Step_{step}",
                inputs={"step": step, "message_history_length": len(messages)}
            )

            response = self.backend.generate(
                messages=messages,
                tools=tool_defs,
                tracer=tracer
            )

            # If the model produced a final textual answer without tool calls
            if not response.tool_calls:
                tracer.end_span(span, outputs={"status": "COMPLETED", "final_answer_length": len(response.content)})
                messages.append(Message.assistant(response.content))
                return AgentExecutionSummary(
                    final_answer=response.content,
                    total_steps=step,
                    tool_calls_executed=tool_calls_count,
                    success=True,
                    conversation_history=messages
                )

            # Process Tool Calls
            messages.append(Message.assistant(content=response.content, tool_calls=response.tool_calls))

            for tc in response.tool_calls:
                tool_signature = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                
                # Cycle / Infinite Loop Detection
                if tool_signature in executed_signatures:
                    err_msg = f"Duplicate Tool Call Detected: '{tc.name}' with identical arguments was already executed. Halting loop."
                    tracer.end_span(span, status="BLOCKED", error=err_msg)
                    return AgentExecutionSummary(
                        final_answer=f"Execution halted due to infinite loop prevention: {err_msg}",
                        total_steps=step,
                        tool_calls_executed=tool_calls_count,
                        success=False,
                        conversation_history=messages,
                        error=err_msg
                    )
                executed_signatures.add(tool_signature)

                # Check for Human-In-The-Loop Confirmation
                tool_obj = self.registry.get(tc.name)
                is_approved = False
                if tool_obj and tool_obj.requires_human_confirmation:
                    if human_approval_callback:
                        is_approved = human_approval_callback(tc)
                    else:
                        is_approved = False

                exec_result = self.executor.execute_tool_call(
                    tool_call=tc,
                    user_permissions=self.user_permissions,
                    human_approved=is_approved
                )
                tool_calls_count += 1

                # Append tool result to context
                messages.append(Message.tool(
                    content=exec_result.to_message_content(),
                    tool_call_id=tc.id,
                    name=tc.name
                ))

            tracer.end_span(span, outputs={"tool_calls_executed_this_step": len(response.tool_calls)})

        # Max steps exceeded
        timeout_err = f"Agent exceeded maximum allowed recursion steps ({self.max_steps})."
        return AgentExecutionSummary(
            final_answer="Process terminated: maximum iteration budget reached.",
            total_steps=self.max_steps,
            tool_calls_executed=tool_calls_count,
            success=False,
            conversation_history=messages,
            error=timeout_err
        )


def demonstrate_tool_loop():
    """Demonstrates multi-turn autonomous tool loop."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.2 AUTONOMOUS MULTI-TURN TOOL LOOP ENGINE ═══[/bold cyan]\n")

    backend = get_llm_backend("mock")
    engine = AgentToolLoopEngine(backend=backend, registry=global_registry, max_steps=5)
    tracer = Tracer("ToolLoop-Demo")

    query = "Check the balance for account ACC_4892 and calculate the volatility of recent risk metrics."
    console.print(f"[bold yellow]Executing Agent Query:[/bold yellow] [white]'{query}'[/white]\n")

    summary = engine.run(user_query=query, tracer=tracer)

    console.print(Panel(summary.final_answer, title="[bold green]Final Agent Synthesis[/bold green]", border_style="green"))
    console.print(f"[dim]Total Steps: {summary.total_steps} | Tool Invocations: {summary.tool_calls_executed}[/dim]\n")

    tracer.render_rich_report(console)


if __name__ == "__main__":
    demonstrate_tool_loop()
