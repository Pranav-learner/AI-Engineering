"""
Layer 1 Phase 5: Master CLI & Interactive Learning Dashboard.
Run this script to interactively explore, benchmark, and test all Phase 5 modules.
"""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from layer_1.phase5.prompt_fundamentals.prompt_structure import demonstrate_prompt_structure
from layer_1.phase5.prompt_fundamentals.prompt_strategies import demonstrate_all_strategies
from layer_1.phase5.prompt_fundamentals.prompt_playground import run_prompt_playground
from layer_1.phase5.prompt_fundamentals.prompt_experiments_and_failures import run_failure_experiments_demo
from layer_1.phase5.structured_outputs_and_tools.structured_outputs import demonstrate_structured_outputs
from layer_1.phase5.structured_outputs_and_tools.tool_schemas_and_dispatch import demonstrate_tool_dispatch
from layer_1.phase5.structured_outputs_and_tools.tool_loop_engine import demonstrate_tool_loop
from layer_1.phase5.structured_outputs_and_tools.tool_simulator_and_failures import run_tool_simulator_demo
from layer_1.phase5.optimization_and_security.prompt_optimization import demonstrate_optimization
from layer_1.phase5.optimization_and_security.prompt_security import demonstrate_security
from layer_1.phase5.optimization_and_security.secure_pipeline import demonstrate_secure_pipeline
from layer_1.phase5.optimization_and_security.attack_experiments import run_attack_benchmark_demo
from layer_1.phase5.full_decision_system import run_full_system_demo


def main():
    console = Console()

    header = """[bold cyan]╔══════════════════════════════════════════════════════════════════════════════╗
║               LAYER 1 PHASE 5 — LLM APPLICATION ENGINEERING                  ║
║      Prompts • Structured Outputs • Tools • Optimization • Security • Traces ║
╚══════════════════════════════════════════════════════════════════════════════╝[/bold cyan]"""

    while True:
        console.print("\n" + header)
        console.print("\n[bold]Select Module or Experiment to Run:[/bold]")
        console.print("  [bold yellow]5.1 Prompt Fundamentals[/bold yellow]")
        console.print("    [1] Prompt Structure & ChatML/XML Assembly Contracts")
        console.print("    [2] Prompt Strategies (Zero/Few-shot, CoT, Self-Consistency, ToT, Planning)")
        console.print("    [3] Interactive Prompt Playground")
        console.print("    [4] Prompt Failure Modes Empirical Benchmark")
        console.print("  [bold yellow]5.2 Structured Outputs & Tools[/bold yellow]")
        console.print("    [5] Structured JSON, Pydantic Validation & Self-Healing Retry Loop")
        console.print("    [6] Tool Registry, Reflection & Sandboxed Dispatch")
        console.print("    [7] Multi-Turn Autonomous Tool Loop Engine")
        console.print("    [8] Tool Failure Simulator & Error Resilience Benchmark")
        console.print("  [bold yellow]5.3 Optimization & Security[/bold yellow]")
        console.print("    [9] Prompt Templates, Compression, Context Window & Versioning")
        console.print("   [10] Prompt Security, Canary Tokens & Defensive Sandboxing")
        console.print("   [11] End-to-End Secure LLM Pipeline")
        console.print("   [12] Red-Team Adversarial Benchmark (15+ Attack Vectors)")
        console.print("  [bold yellow]Full Application & Verification[/bold yellow]")
        console.print("   [13] 🏛️ Autonomous Financial Risk & Operations Decision Engine (AFRO-DE)")
        console.print("   [14] 🧪 Run All Modules in Sequence")
        console.print("   [0] Exit\n")

        if not sys.stdin.isatty():
            console.print("[yellow]Non-interactive mode detected. Executing all modules in sequence...[/yellow]")
            choice = "14"
        else:
            try:
                choice = Prompt.ask("Enter selection", choices=[str(i) for i in range(15)], default="13")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting.[/dim]")
                break

        if choice == "0":
            console.print("[dim]Goodbye![/dim]")
            break
        elif choice == "1":
            demonstrate_prompt_structure()
        elif choice == "2":
            demonstrate_all_strategies()
        elif choice == "3":
            run_prompt_playground(interactive=sys.stdin.isatty())
        elif choice == "4":
            run_failure_experiments_demo()
        elif choice == "5":
            demonstrate_structured_outputs()
        elif choice == "6":
            demonstrate_tool_dispatch()
        elif choice == "7":
            demonstrate_tool_loop()
        elif choice == "8":
            run_tool_simulator_demo()
        elif choice == "9":
            demonstrate_optimization()
        elif choice == "10":
            demonstrate_security()
        elif choice == "11":
            demonstrate_secure_pipeline()
        elif choice == "12":
            run_attack_benchmark_demo()
        elif choice == "13":
            run_full_system_demo()
        elif choice == "14":
            console.print("\n[bold cyan]═══ EXECUTING COMPLETE PHASE 5 TEST AND BENCHMARK SUITE ═══[/bold cyan]\n")
            demonstrate_prompt_structure()
            demonstrate_all_strategies()
            run_failure_experiments_demo()
            demonstrate_structured_outputs()
            demonstrate_tool_dispatch()
            demonstrate_tool_loop()
            run_tool_simulator_demo()
            demonstrate_optimization()
            demonstrate_security()
            demonstrate_secure_pipeline()
            run_attack_benchmark_demo()
            run_full_system_demo()
            console.print("\n[bold green]✔ All 12 Phase 5 demonstrations and benchmarks completed successfully![/bold green]\n")
            if not sys.stdin.isatty():
                break

        if sys.stdin.isatty():
            input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    main()
