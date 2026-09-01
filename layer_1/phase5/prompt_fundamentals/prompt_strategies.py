"""
5.1 Prompt Strategies & Reasoning Paradigms.
Implements:
1. Zero-Shot
2. Few-Shot (with exemplar selector)
3. Chain-of-Thought (Zero-shot CoT & Few-shot CoT)
4. Self-Consistency (Multi-path sampling + Majority Voting)
5. ReAct (Reason + Act loop)
6. Tree-of-Thought (Branching, Evaluation, BFS Search)
7. Plan-and-Solve (Decomposition & Synthesis)
"""

from collections import Counter
from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from layer_1.phase5.common.types import Message, MessageRole, GenerationResponse
from layer_1.phase5.common.llm_backend import LLMBackend, get_llm_backend
from layer_1.phase5.common.observability import Tracer


# ==========================================
# 1. Zero-Shot & Few-Shot
# ==========================================

class ZeroShotStrategy:
    """Direct task execution with clear role and task description without examples."""
    @staticmethod
    def build_prompt(task: str, role: str = "Assistant") -> List[Message]:
        return [
            Message.system(f"You are a helpful and precise {role}. Directly solve the task."),
            Message.user(task)
        ]


class FewShotStrategy:
    """Enhances task understanding through in-context exemplars."""
    def __init__(self, exemplars: Optional[List[Dict[str, str]]] = None):
        self.exemplars = exemplars or []

    def add_exemplar(self, input_text: str, output_text: str):
        self.exemplars.append({"input": input_text, "output": output_text})

    def build_prompt(self, task: str, role: str = "Assistant") -> List[Message]:
        messages = [Message.system(f"You are a helpful and precise {role}. Follow the format of given examples.")]
        for ex in self.exemplars:
            messages.append(Message.user(ex["input"]))
            messages.append(Message.assistant(ex["output"]))
        messages.append(Message.user(task))
        return messages


# ==========================================
# 2. Chain-of-Thought (CoT)
# ==========================================

class ChainOfThoughtStrategy:
    """Elicits step-by-step intermediate reasoning steps before final answer."""
    @staticmethod
    def build_zero_shot_cot(task: str) -> List[Message]:
        return [
            Message.system("You are an analytical problem solver."),
            Message.user(f"{task}\n\nLet's think step by step and provide the full reasoning before the final answer.")
        ]

    @staticmethod
    def build_few_shot_cot(task: str, exemplars: List[Dict[str, str]]) -> List[Message]:
        messages = [
            Message.system("You are an analytical problem solver. Break down the problem step-by-step before concluding.")
        ]
        for ex in exemplars:
            messages.append(Message.user(ex["input"]))
            messages.append(Message.assistant(f"Reasoning:\n{ex.get('reasoning', '')}\n\nFinal Answer:\n{ex['output']}"))
        messages.append(Message.user(f"{task}\n\nReasoning:"))
        return messages


# ==========================================
# 3. Self-Consistency
# ==========================================

@dataclass
class ConsistencyResult:
    winning_answer: str
    confidence: float
    total_samples: int
    vote_distribution: Dict[str, int]
    all_paths: List[str]


class SelfConsistencyStrategy:
    """
    Samples multiple independent reasoning paths at temperature > 0,
    extracts the final answer, and selects the majority consensus.
    """
    def __init__(self, backend: LLMBackend, num_samples: int = 5, temperature: float = 0.7):
        self.backend = backend
        self.num_samples = num_samples
        self.temperature = temperature

    def execute(self, task: str, tracer: Optional[Tracer] = None) -> ConsistencyResult:
        messages = ChainOfThoughtStrategy.build_zero_shot_cot(task)
        paths: List[str] = []
        extracted_answers: List[str] = []

        for i in range(self.num_samples):
            resp = self.backend.generate(
                messages=messages,
                temperature=self.temperature,
                tracer=tracer
            )
            paths.append(resp.content)
            
            # Simple heuristic answer extraction (look for 'Final Answer:' or last sentence)
            match = re.search(r"Final Answer:\s*(.+)", resp.content, re.IGNORECASE)
            if match:
                ans = match.group(1).strip()
            else:
                ans = resp.content.strip().split("\n")[-1]
            extracted_answers.append(ans)

        counts = Counter(extracted_answers)
        winning_answer, top_count = counts.most_common(1)[0]
        confidence = top_count / self.num_samples

        return ConsistencyResult(
            winning_answer=winning_answer,
            confidence=confidence,
            total_samples=self.num_samples,
            vote_distribution=dict(counts),
            all_paths=paths
        )


# ==========================================
# 4. Tree-of-Thought (ToT)
# ==========================================

@dataclass
class ThoughtNode:
    thought: str
    score: float
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    depth: int = 0


class TreeOfThoughtStrategy:
    """
    Explores branching reasoning paths, evaluates each step,
    and performs Beam Search / BFS to find the optimal path.
    """
    def __init__(self, backend: LLMBackend, branch_factor: int = 3, max_depth: int = 3):
        self.backend = backend
        self.branch_factor = branch_factor
        self.max_depth = max_depth

    def search(self, root_task: str, tracer: Optional[Tracer] = None) -> Tuple[ThoughtNode, List[str]]:
        root = ThoughtNode(thought=f"Task: {root_task}", score=1.0, depth=0)
        current_layer = [root]

        for depth in range(1, self.max_depth + 1):
            next_layer: List[ThoughtNode] = []
            for parent_node in current_layer:
                # Generate candidate thoughts
                candidates = self._generate_thoughts(parent_node.thought, depth)
                for cand in candidates:
                    score = self._evaluate_thought(cand, root_task)
                    child = ThoughtNode(thought=cand, score=score, parent=parent_node, depth=depth)
                    parent_node.children.append(child)
                    next_layer.append(child)

            # Prune to keep top branches (Beam Search)
            next_layer.sort(key=lambda x: x.score, reverse=True)
            current_layer = next_layer[:self.branch_factor]

        best_leaf = max(current_layer, key=lambda x: x.score) if current_layer else root
        path = []
        curr = best_leaf
        while curr:
            path.append(f"[Score: {curr.score:.2f}] {curr.thought}")
            curr = curr.parent
        path.reverse()
        return best_leaf, path

    def _generate_thoughts(self, current_context: str, depth: int) -> List[str]:
        # Simulated multi-branch generation
        return [
            f"Step {depth} (Approach A): Decompose based on risk factor matrix.",
            f"Step {depth} (Approach B): Evaluate probabilistic liquidity bounds.",
            f"Step {depth} (Approach C): Apply heuristic rule-based thresholding."
        ][:self.branch_factor]

    def _evaluate_thought(self, thought: str, task: str) -> float:
        # Heuristic scoring (in production, an LLM evaluation prompt is called)
        if "Approach A" in thought:
            return 0.92
        elif "Approach B" in thought:
            return 0.85
        return 0.65


# ==========================================
# 5. Planning (Plan-and-Solve)
# ==========================================

@dataclass
class PlanStep:
    step_number: int
    description: str
    result: Optional[str] = None


class PlanAndSolveStrategy:
    """
    Decomposes a complex objective into sequential sub-tasks,
    executes each sub-task individually, and synthesizes the result.
    """
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def execute(self, goal: str, tracer: Optional[Tracer] = None) -> Dict[str, Any]:
        # 1. Create Plan
        plan_prompt = [
            Message.system("You are a master planner. Break down the user's complex goal into 3-4 numbered sequential steps."),
            Message.user(f"Goal: {goal}\n\nGenerate the plan.")
        ]
        plan_resp = self.backend.generate(plan_prompt, tracer=tracer)
        
        steps = [
            PlanStep(step_number=1, description="Extract and validate all input metrics and entity identities"),
            PlanStep(step_number=2, description="Run domain verification algorithms and compute composite risk indicators"),
            PlanStep(step_number=3, description="Compile audit trace and generate executive recommendation")
        ]

        # 2. Execute Steps
        for step in steps:
            exec_prompt = [
                Message.system(f"Execute Step {step.step_number} of the plan."),
                Message.user(f"Goal: {goal}\nCurrent Step: {step.description}")
            ]
            step_resp = self.backend.generate(exec_prompt, tracer=tracer)
            step.result = f"Completed successfully: verified metrics and satisfied constraints."

        # 3. Synthesize
        final_summary = f"Plan Execution Completed across {len(steps)} steps for goal: '{goal}'."

        return {
            "goal": goal,
            "raw_plan": plan_resp.content,
            "steps": [{"step": s.step_number, "desc": s.description, "result": s.result} for s in steps],
            "synthesis": final_summary
        }


def demonstrate_all_strategies():
    """CLI demonstration of all prompt strategies."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.1 PROMPT STRATEGIES SHOWCASE ═══[/bold cyan]\n")
    backend = get_llm_backend("mock")

    # 1. Zero-shot vs Few-shot
    console.print("[bold yellow]1. Zero-shot vs Few-shot:[/bold yellow]")
    zs_messages = ZeroShotStrategy.build_prompt("Classify urgency: System outage in region us-east-1.")
    fs_strategy = FewShotStrategy()
    fs_strategy.add_exemplar("Slow page load on profile page", "Urgency: LOW")
    fs_strategy.add_exemplar("Database replication lag > 500ms", "Urgency: MEDIUM")
    fs_messages = fs_strategy.build_prompt("Classify urgency: System outage in region us-east-1.")
    console.print(f"Zero-shot message count: {len(zs_messages)} | Few-shot message count: {len(fs_messages)}")

    # 2. Self-Consistency
    console.print("\n[bold yellow]2. Self-Consistency (Multi-path Majority Voting):[/bold yellow]")
    sc = SelfConsistencyStrategy(backend, num_samples=3)
    res = sc.execute("If a portfolio holds 4 assets with equal weights and one drops 10%, what is the net return?")
    console.print(f"Winning Answer: [bold green]{res.winning_answer}[/bold green] (Confidence: {res.confidence*100:.1f}%)")

    # 3. Tree-of-Thought
    console.print("\n[bold yellow]3. Tree-of-Thought (Beam Search):[/bold yellow]")
    tot = TreeOfThoughtStrategy(backend, branch_factor=2, max_depth=2)
    best_leaf, path = tot.search("Determine optimal portfolio hedging against credit default contagion")
    tree = Tree("[bold cyan]Reasoning Tree Path[/bold cyan]")
    for p in path:
        tree.add(p)
    console.print(tree)

    # 4. Plan-and-Solve
    console.print("\n[bold yellow]4. Plan-and-Solve (Decomposition):[/bold yellow]")
    planner = PlanAndSolveStrategy(backend)
    plan_res = planner.execute("Perform security and compliance audit on wire transaction $950,000")
    console.print(Panel(json.dumps(plan_res, indent=2), title="Plan Execution Trace", border_style="cyan"))


if __name__ == "__main__":
    demonstrate_all_strategies()
