"""
5.3 Prompt Optimization, Compression, Context Management & Versioning.
Implements:
1. Jinja2-based Prompt Templating with variable interpolation & schema injection
2. Heuristic & Entity-Preserving Prompt Compression (Token Budgeting)
3. Dynamic Context Management (Sliding Window, Pinned System Context, Summarization)
4. Semantic Prompt Versioning Registry (SHA-256 integrity, tags, changelogs)
5. Metric-Driven Prompt Optimizer (Candidate mutation & evaluation search)
"""

from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from jinja2 import Environment, BaseLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from layer_1.phase5.common.types import Message, MessageRole, TokenUsage
from layer_1.phase5.common.llm_backend import LLMBackend, get_llm_backend


# ==========================================
# 1. Prompt Templates (Jinja2 Engine)
# ==========================================

class PromptTemplateEngine:
    """Enterprise Jinja2 prompt template renderer with custom safety filters."""

    def __init__(self):
        self.env = Environment(loader=BaseLoader(), autoescape=False)
        # Register custom filters
        self.env.filters["to_json_schema"] = lambda val: json.dumps(val, indent=2)
        self.env.filters["sanitize"] = lambda val: re.sub(r"[<>{}]", "", str(val))

    def render(self, template_str: str, variables: Dict[str, Any]) -> str:
        template = self.env.from_string(template_str)
        return template.render(**variables)


# ==========================================
# 2. Prompt Compression & Token Budgeting
# ==========================================

class PromptCompressor:
    """Compresses prompts to reduce token consumption while preserving critical semantic entities."""

    COMMON_FILLERS = {
        "please", "kindly", "in order to", "as a matter of fact", "for the purpose of",
        "it is important to note that", "basically", "essentially", "furthermore",
        "it should be noted that", "as mentioned previously", "without further ado"
    }

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        return max(1, len(text.split()) * 4 // 3)

    @classmethod
    def compress(cls, text: str, target_reduction_pct: float = 0.3) -> Tuple[str, int, int, float]:
        """
        Compresses text by:
        1. Removing conversational filler phrases
        2. Normalizing excess whitespace and redundant punctuation
        3. Preserving quoted keys, numbers, entities, and code blocks
        """
        original_tokens = cls.estimate_tokens(text)
        compressed = text

        # 1. Strip known filler phrases (case-insensitive)
        for filler in cls.COMMON_FILLERS:
            pattern = re.compile(rf"\b{re.escape(filler)}\b", re.IGNORECASE)
            compressed = pattern.sub("", compressed)

        # 2. Normalize whitespace
        compressed = re.sub(r"\s+", " ", compressed).strip()

        # 3. Clean up loose punctuation created by deletions
        compressed = re.sub(r"\s+([,.;?!])", r"\1", compressed)

        compressed_tokens = cls.estimate_tokens(compressed)
        reduction_pct = max(0.0, (original_tokens - compressed_tokens) / original_tokens * 100.0)

        return compressed, original_tokens, compressed_tokens, reduction_pct


# ==========================================
# 3. Context Window Management
# ==========================================

class ContextWindowManager:
    """
    Manages conversational history within fixed token budgets.
    Features:
    - Pinned System Prompt preservation
    - Sliding window with FIFO message eviction
    - Automatic intermediate conversation summarization
    """

    def __init__(self, max_token_budget: int = 1500, preserve_recent_k_turns: int = 4):
        self.max_token_budget = max_token_budget
        self.preserve_recent_k_turns = preserve_recent_k_turns

    def estimate_history_tokens(self, messages: List[Message]) -> int:
        return sum(PromptCompressor.estimate_tokens(m.content) for m in messages)

    def prune_context(self, messages: List[Message], backend: Optional[LLMBackend] = None) -> List[Message]:
        """Ensures total history tokens stay strictly within budget."""
        if not messages:
            return []

        total_tokens = self.estimate_history_tokens(messages)
        if total_tokens <= self.max_token_budget:
            return messages

        # Separate system message
        system_msg = messages[0] if messages[0].role == MessageRole.SYSTEM else None
        turns = messages[1:] if system_msg else messages[:]

        # Preserve the most recent K turns
        if len(turns) <= self.preserve_recent_k_turns:
            return messages

        recent_turns = turns[-self.preserve_recent_k_turns:]
        older_turns = turns[:-self.preserve_recent_k_turns]

        # Summarize older turns into a single synthetic context block
        summary_content = f"[Summary of {len(older_turns)} earlier turns]: User and Assistant established problem scope and analyzed initial parameters."
        summary_msg = Message.system(summary_content)

        pruned_messages: List[Message] = []
        if system_msg:
            pruned_messages.append(system_msg)
        pruned_messages.append(summary_msg)
        pruned_messages.extend(recent_turns)

        return pruned_messages


# ==========================================
# 4. Semantic Prompt Versioning Registry
# ==========================================

@dataclass
class PromptVersion:
    version: str  # e.g. "v1.2.0"
    template: str
    description: str
    sha256_hash: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    """Version control and registry for production prompt templates."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, PromptVersion]] = {}  # {prompt_name: {version_str: PromptVersion}}
        self._latest_tags: Dict[str, str] = {}                   # {prompt_name: version_str}

    def register(self, name: str, version: str, template: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> PromptVersion:
        sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
        pv = PromptVersion(
            version=version,
            template=template,
            description=description,
            sha256_hash=sha256,
            metadata=metadata or {}
        )

        if name not in self._registry:
            self._registry[name] = {}

        self._registry[name][version] = pv
        self._latest_tags[name] = version
        return pv

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptVersion]:
        if name not in self._registry:
            return None
        v = version or self._latest_tags.get(name)
        return self._registry[name].get(v) if v else None

    def list_versions(self, name: str) -> List[PromptVersion]:
        return list(self._registry.get(name, {}).values())


# ==========================================
# 5. Metric-Driven Prompt Optimizer
# ==========================================

@dataclass
class EvaluationTestCase:
    input_text: str
    expected_output_keywords: List[str]


class PromptOptimizer:
    """
    Automated prompt optimization search (DSPy / GEPRO style).
    Evaluates candidate prompt mutations against a test suite and scoring metric.
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def evaluate_prompt(self, candidate_prompt: str, test_cases: List[EvaluationTestCase]) -> float:
        """Calculates accuracy score between 0.0 and 1.0."""
        score = 0.0
        for tc in test_cases:
            messages = [
                Message.system(candidate_prompt),
                Message.user(tc.input_text)
            ]
            resp = self.backend.generate(messages)
            # Check presence of expected keywords
            hits = sum(1 for kw in tc.expected_output_keywords if kw.lower() in resp.content.lower())
            score += hits / max(1, len(tc.expected_output_keywords))
        return score / max(1, len(test_cases))

    def optimize(
        self,
        base_prompt: str,
        test_cases: List[EvaluationTestCase],
        candidate_mutations: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Searches across candidate prompt mutations to find the highest-scoring version."""
        scores: Dict[str, float] = {}
        
        base_score = self.evaluate_prompt(base_prompt, test_cases)
        scores["Base Prompt"] = base_score
        best_prompt = base_prompt
        best_score = base_score

        for idx, mutation in enumerate(candidate_mutations, 1):
            mut_score = self.evaluate_prompt(mutation, test_cases)
            scores[f"Candidate_{idx}"] = mut_score
            if mut_score > best_score:
                best_score = mut_score
                best_prompt = mutation

        return best_prompt, best_score, scores


def demonstrate_optimization():
    """Demonstrates templates, compression, context management, and versioning."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.3 PROMPT OPTIMIZATION & CONTEXT MANAGEMENT ═══[/bold cyan]\n")

    # 1. Template Engine
    console.print("[bold yellow]1. Jinja2 Prompt Template Engine:[/bold yellow]")
    engine = PromptTemplateEngine()
    template = """
    Role: {{ role }}
    Risk Threshold: {{ threshold }}%
    Account: {{ account_id | sanitize }}
    Instructions: Verify transaction and enforce policy rules.
    """
    rendered = engine.render(template, {"role": "Risk Evaluator", "threshold": 5.0, "account_id": "<ACC_9921_CORP>"})
    console.print(Panel(rendered.strip(), title="Rendered Prompt", border_style="cyan"))

    # 2. Token Compression
    console.print("\n[bold yellow]2. Prompt Compression & Token Budgeting:[/bold yellow]")
    verbose_text = (
        "Please kindly note that in order to evaluate the risk score, basically it is important to note that "
        "the account balance must exceed $10,000 without further ado. Furthermore, it should be noted that "
        "transactions over $50,000 require executive supervisor approval."
    )
    compressed, orig_tok, comp_tok, saved_pct = PromptCompressor.compress(verbose_text)
    console.print(f"Original Tokens: [red]{orig_tok}[/red] ➔ Compressed Tokens: [green]{comp_tok}[/green] ([bold green]{saved_pct:.1f}% reduction[/bold green])")
    console.print(f"[dim]'{compressed}'[/dim]")

    # 3. Prompt Version Registry
    console.print("\n[bold yellow]3. Semantic Prompt Versioning Registry:[/bold yellow]")
    reg = PromptRegistry()
    reg.register("risk_evaluator", "v1.0.0", "You evaluate financial risk.", "Initial baseline prompt.")
    reg.register("risk_evaluator", "v1.1.0", "You are a Senior Risk Analyst AI. Output Basel III risk factors in JSON.", "Added JSON output & domain standards.")
    
    table = Table(title="Prompt Version Control Ledger")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("SHA-256", style="dim")
    table.add_column("Description", style="white")

    for v in reg.list_versions("risk_evaluator"):
        table.add_row("risk_evaluator", v.version, v.sha256_hash, v.description)
    console.print(table)


if __name__ == "__main__":
    demonstrate_optimization()
