"""
Observability, tracing, and metric collection for LLM engineering.
"""

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel

from layer_1.phase5.common.types import TokenUsage


@dataclass
class Span:
    """Individual unit of work within an LLM execution trace."""
    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "RUNNING"  # "OK", "ERROR", "BLOCKED"
    error_message: Optional[str] = None

    def end(self, status: str = "OK", error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status
        self.error_message = error


class Tracer:
    """Global/Contextual tracer for logging LLM steps, tool calls, and guardrail decisions."""

    def __init__(self, trace_name: str = "LLM-Execution"):
        self.trace_name = trace_name
        self.spans: List[Span] = []
        self.active_spans: List[Span] = []
        self._counter = 0

    def start_span(self, name: str, inputs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Span:
        self._counter += 1
        span_id = f"span_{self._counter:03d}"
        parent_id = self.active_spans[-1].span_id if self.active_spans else None
        span = Span(
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            inputs=inputs or {},
            metadata=metadata or {}
        )
        self.spans.append(span)
        self.active_spans.append(span)
        return span

    def end_span(self, span: Span, outputs: Optional[Dict[str, Any]] = None, usage: Optional[TokenUsage] = None, status: str = "OK", error: Optional[str] = None):
        if outputs:
            span.outputs = outputs
        if usage:
            span.usage = usage
        span.end(status=status, error=error)
        if self.active_spans and self.active_spans[-1].span_id == span.span_id:
            self.active_spans.pop()

    @property
    def total_usage(self) -> TokenUsage:
        tot = TokenUsage()
        for s in self.spans:
            tot = tot.add(s.usage)
        return tot

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time or s.start_time for s in self.spans)
        return (end - start) * 1000.0

    def render_rich_report(self, console: Optional[Console] = None):
        console = console or Console()
        
        # Summary Table
        table = Table(title=f"📊 Trace Summary: [bold cyan]{self.trace_name}[/bold cyan]")
        table.add_column("Span ID", style="dim", width=10)
        table.add_column("Operation", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Tokens (P/C/Tot)", justify="right")
        table.add_column("Est. Cost ($)", justify="right")

        for s in self.spans:
            status_style = "green" if s.status == "OK" else ("red" if s.status == "ERROR" else "yellow")
            tokens_str = f"{s.usage.prompt_tokens}/{s.usage.completion_tokens}/{s.usage.total_tokens}"
            cost_str = f"${s.usage.estimated_cost_usd:.5f}" if s.usage.estimated_cost_usd > 0 else "-"
            table.add_row(
                s.span_id,
                f"{'  ' if s.parent_id else ''}{s.name}",
                f"[{status_style}]{s.status}[/{status_style}]",
                f"{s.duration_ms:.1f}",
                tokens_str,
                cost_str
            )

        console.print(table)
        
        # Overall Stats
        tot = self.total_usage
        stats_panel = Panel(
            f"[bold]Total Latency:[/bold] {self.total_duration_ms:.1f} ms | "
            f"[bold]Total Spans:[/bold] {len(self.spans)} | "
            f"[bold]Prompt Tokens:[/bold] {tot.prompt_tokens} | "
            f"[bold]Completion Tokens:[/bold] {tot.completion_tokens} | "
            f"[bold]Total Tokens:[/bold] {tot.total_tokens} | "
            f"[bold]Total Cost:[/bold] ${tot.estimated_cost_usd:.5f}",
            title="Trace Metrics",
            border_style="cyan"
        )
        console.print(stats_panel)
