"""
5.1 Interactive Prompt Playground.
Allows engineers to interactively test and compare prompt strategies,
inspect compiled tokens, analyze latency, and evaluate output quality.
"""

import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from layer_1.phase5.common.types import Message, MessageRole
from layer_1.phase5.common.llm_backend import get_llm_backend
from layer_1.phase5.common.observability import Tracer
from layer_1.phase5.prompt_fundamentals.prompt_structure import StructuredPrompt, OutputConstraint
from layer_1.phase5.prompt_fundamentals.prompt_strategies import (
    ZeroShotStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
    SelfConsistencyStrategy,
    TreeOfThoughtStrategy,
    PlanAndSolveStrategy,
)


def run_prompt_playground(interactive: bool = True):
    console = Console()
    console.print("\n[bold cyan]═════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       🧪 INTERACTIVE PROMPT PLAYGROUND (PHASE 5.1)         [/bold cyan]")
    console.print("[bold cyan]═════════════════════════════════════════════════════════════[/bold cyan]\n")

    backend = get_llm_backend("mock")
    tracer = Tracer("Playground-Trace")

    sample_task = "Evaluate if account ACC_901 with balance $12,000 can process a $15,000 transaction with $5,000 overdraft limit."

    if not interactive:
        console.print(f"[bold yellow]Running automated demonstration with default task:[/bold yellow] [white]{sample_task}[/white]\n")
        choice = "1"
    else:
        console.print("[bold]Select Reasoning / Prompting Strategy to Test:[/bold]")
        console.print("  [1] Structured Prompt (System + Context + User + Constraints)")
        console.print("  [2] Zero-Shot vs Few-Shot Comparison")
        console.print("  [3] Chain-of-Thought (Step-by-step intermediate reasoning)")
        console.print("  [4] Self-Consistency (Sample N paths + Majority vote)")
        console.print("  [5] Tree-of-Thought (Branching thought search)")
        console.print("  [6] Plan-and-Solve (Task decomposition & step execution)")
        console.print("  [0] Exit")
        
        try:
            choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6"], default="1")
        except (EOFError, KeyboardInterrupt):
            choice = "1"

    if choice == "0":
        console.print("[dim]Exiting playground.[/dim]")
        return

    # Task 1: Structured Prompt
    if choice == "1":
        prompt = StructuredPrompt(
            system_instruction="You are an enterprise financial risk compliance engine."
        )
        prompt.add_context("Account Data", "ACC_901: Balance=$12,000 | Overdraft Limit=$5,000 | Active Status=True")
        prompt.set_user_instruction(sample_task)
        prompt.set_output_constraint(OutputConstraint(format_type="json", json_schema={"approved": "bool", "remaining_credit": "float", "rationale": "string"}))
        
        messages = prompt.compile_messages()
        resp = backend.generate(messages, tracer=tracer)
        
        console.print(Panel(resp.content, title="[bold green]Structured Output[/bold green]", border_style="green"))
        tracer.render_rich_report(console)

    # Task 2: Few-Shot Comparison
    elif choice == "2":
        fs = FewShotStrategy()
        fs.add_exemplar(
            "Account $1,000, Overdraft $0, Request $1,500",
            "Result: REJECTED (Insufficient funds by $500)"
        )
        fs.add_exemplar(
            "Account $5,000, Overdraft $2,000, Request $6,000",
            "Result: APPROVED (Balance drops to -$1,000, within $2,000 overdraft limit)"
        )
        messages = fs.build_prompt(sample_task)
        resp = backend.generate(messages, tracer=tracer)
        console.print(Panel(resp.content, title="[bold green]Few-Shot Response[/bold green]", border_style="green"))

    # Task 3: Chain-of-Thought
    elif choice == "3":
        messages = ChainOfThoughtStrategy.build_zero_shot_cot(sample_task)
        resp = backend.generate(messages, tracer=tracer)
        console.print(Panel(resp.content, title="[bold green]Chain-of-Thought Reasoning[/bold green]", border_style="green"))

    # Task 4: Self-Consistency
    elif choice == "4":
        sc = SelfConsistencyStrategy(backend, num_samples=3)
        res = sc.execute(sample_task, tracer=tracer)
        table = Table(title="Self-Consistency Aggregation")
        table.add_column("Candidate Conclusion", style="cyan")
        table.add_column("Votes", justify="center", style="bold magenta")
        for ans, count in res.vote_distribution.items():
            table.add_row(ans, str(count))
        console.print(table)
        console.print(f"[bold green]Winning Consensus:[/bold green] {res.winning_answer} (Confidence: {res.confidence*100:.0f}%)")

    # Task 5: Tree-of-Thought
    elif choice == "5":
        tot = TreeOfThoughtStrategy(backend, branch_factor=2, max_depth=2)
        leaf, path = tot.search(sample_task, tracer=tracer)
        console.print("[bold green]Optimal Tree-of-Thought Trajectory:[/bold green]")
        for step in path:
            console.print(f"  ➜ {step}")

    # Task 6: Plan and Solve
    elif choice == "6":
        planner = PlanAndSolveStrategy(backend)
        plan_out = planner.execute(sample_task, tracer=tracer)
        console.print(Panel(str(plan_out["synthesis"]), title="[bold green]Plan-and-Solve Result[/bold green]", border_style="green"))

    console.print("\n[bold green]✔ Playground execution completed successfully.[/bold green]\n")


if __name__ == "__main__":
    interactive_mode = sys.stdin.isatty()
    run_prompt_playground(interactive=interactive_mode)
