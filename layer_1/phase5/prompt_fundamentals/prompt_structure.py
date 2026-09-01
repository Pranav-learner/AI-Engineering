"""
5.1 Prompt Structure & Composition.
Demonstrates how professional LLM engineering structures prompts into modular,
enforceable components:
- System Instructions (Persona, Goals, Safety Policies)
- Context Blocks (Retrieved documents, User profile, Session state)
- User Instructions (Specific task / goal)
- Output Requirements (Format, schema, constraints, forbidden tokens)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from layer_1.phase5.common.types import Message, MessageRole


class PromptFormat(str, Enum):
    CHAT_MESSAGES = "chat_messages"  # List of Message objects (Standard API)
    CHAT_ML = "chat_ml"              # <|im_start|>system...<|im_end|>
    XML_WRAPPED = "xml_wrapped"      # <system>...</system><context>...</context>
    MARKDOWN = "markdown"            # ## System\n## Context\n## Task


@dataclass
class ContextBlock:
    """Represents a chunk of retrieved or injected contextual data."""
    title: str
    content: str
    source: Optional[str] = None
    priority: int = 1  # Lower number = higher priority when truncating

    def to_xml(self) -> str:
        src_attr = f' source="{self.source}"' if self.source else ""
        return f'<document title="{self.title}"{src_attr}>\n{self.content}\n</document>'

    def to_markdown(self) -> str:
        src_info = f" (Source: {self.source})" if self.source else ""
        return f"### Document: {self.title}{src_info}\n```\n{self.content}\n```"


@dataclass
class OutputConstraint:
    """Defines strict contractual constraints on LLM output."""
    format_type: str = "json"  # "json", "markdown", "yaml", "plaintext"
    json_schema: Optional[Dict[str, Any]] = None
    max_length_words: Optional[int] = None
    forbidden_phrases: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)

    def render_instruction(self) -> str:
        instructions = ["## Output Requirements & Constraints:"]
        if self.format_type.lower() == "json":
            instructions.append("1. Respond ONLY with valid, parseable JSON conforming to the schema.")
            instructions.append("2. Do not wrap JSON in explanatory text, preamble, or conversational fluff.")
            if self.json_schema:
                import json
                instructions.append(f"3. Strict JSON Schema:\n```json\n{json.dumps(self.json_schema, indent=2)}\n```")
        elif self.format_type.lower() == "markdown":
            instructions.append("1. Format the response using clean GitHub Flavored Markdown.")
        
        if self.max_length_words:
            instructions.append(f"- Strictly keep the entire response under {self.max_length_words} words.")
        if self.forbidden_phrases:
            instructions.append(f"- Do NOT use the following phrases: {', '.join(self.forbidden_phrases)}.")
        if self.required_sections:
            instructions.append(f"- You must include these exact headings: {', '.join(self.required_sections)}.")
            
        return "\n".join(instructions)


class StructuredPrompt:
    """
    Builder and compiler for structured enterprise prompts.
    Enforces clean separation between System, Context, User Goal, and Output Contract.
    """

    def __init__(self, system_instruction: str = ""):
        self.system_instruction = system_instruction
        self.context_blocks: List[ContextBlock] = []
        self.user_instruction: str = ""
        self.output_constraint: Optional[OutputConstraint] = None
        self.exemplars: List[Dict[str, str]] = []  # Few-shot examples [{"input": "...", "output": "..."}]

    def set_system_instruction(self, instruction: str) -> "StructuredPrompt":
        self.system_instruction = instruction
        return self

    def add_context(self, title: str, content: str, source: Optional[str] = None, priority: int = 1) -> "StructuredPrompt":
        self.context_blocks.append(ContextBlock(title=title, content=content, source=source, priority=priority))
        return self

    def add_exemplar(self, user_example: str, assistant_example: str) -> "StructuredPrompt":
        self.exemplars.append({"input": user_example, "output": assistant_example})
        return self

    def set_user_instruction(self, instruction: str) -> "StructuredPrompt":
        self.user_instruction = instruction
        return self

    def set_output_constraint(self, constraint: OutputConstraint) -> "StructuredPrompt":
        self.output_constraint = constraint
        return self

    def compile_messages(self) -> List[Message]:
        """Compiles the structured components into a standard list of Chat Messages."""
        messages: List[Message] = []

        # 1. System Prompt (Role + Behavioral policy + Output Constraints)
        full_system_parts = []
        if self.system_instruction:
            full_system_parts.append(self.system_instruction)
        
        if self.output_constraint:
            full_system_parts.append(self.output_constraint.render_instruction())

        if full_system_parts:
            messages.append(Message.system("\n\n".join(full_system_parts)))

        # 2. Few-shot Exemplars (if any)
        for ex in self.exemplars:
            messages.append(Message.user(ex["input"]))
            messages.append(Message.assistant(ex["output"]))

        # 3. Context & User Instruction (in User turn or separated turns)
        user_body_parts = []
        if self.context_blocks:
            user_body_parts.append("<context>")
            for cb in self.context_blocks:
                user_body_parts.append(cb.to_xml())
            user_body_parts.append("</context>\n")

        user_body_parts.append(f"<task>\n{self.user_instruction}\n</task>")
        messages.append(Message.user("\n".join(user_body_parts)))

        return messages

    def compile_text(self, format_type: PromptFormat = PromptFormat.XML_WRAPPED) -> str:
        """Compiles prompt into a single formatted string for raw completion engines."""
        if format_type == PromptFormat.CHAT_ML:
            lines = [f"<|im_start|>system\n{self.system_instruction}"]
            if self.output_constraint:
                lines.append(self.output_constraint.render_instruction())
            lines.append("<|im_end|>")
            for ex in self.exemplars:
                lines.append(f"<|im_start|>user\n{ex['input']}<|im_end|>")
                lines.append(f"<|im_start|>assistant\n{ex['output']}<|im_end|>")
            lines.append("<|im_start|>user")
            if self.context_blocks:
                lines.append("<context>")
                for cb in self.context_blocks:
                    lines.append(cb.to_xml())
                lines.append("</context>")
            lines.append(f"<task>{self.user_instruction}</task><|im_end|>")
            lines.append("<|im_start|>assistant")
            return "\n".join(lines)

        elif format_type == PromptFormat.MARKDOWN:
            parts = [f"# SYSTEM INSTRUCTION\n{self.system_instruction}\n"]
            if self.context_blocks:
                parts.append("# CONTEXT")
                for cb in self.context_blocks:
                    parts.append(cb.to_markdown())
                parts.append("")
            if self.exemplars:
                parts.append("# EXAMPLES")
                for idx, ex in enumerate(self.exemplars, 1):
                    parts.append(f"**Example {idx} Input:** {ex['input']}\n**Example {idx} Output:** {ex['output']}\n")
            parts.append(f"# USER TASK\n{self.user_instruction}\n")
            if self.output_constraint:
                parts.append(self.output_constraint.render_instruction())
            return "\n".join(parts)

        # Default XML_WRAPPED
        parts = [f"<system>\n{self.system_instruction}\n</system>"]
        if self.context_blocks:
            parts.append("<context>")
            for cb in self.context_blocks:
                parts.append(cb.to_xml())
            parts.append("</context>")
        parts.append(f"<user_query>\n{self.user_instruction}\n</user_query>")
        if self.output_constraint:
            parts.append(f"<output_requirements>\n{self.output_constraint.render_instruction()}\n</output_requirements>")
        return "\n\n".join(parts)


def demonstrate_prompt_structure():
    """Interactive demo showcasing structured prompt assembly and rendering."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.1 PROMPT FUNDAMENTALS: STRUCTURE & CONTRACTS ═══[/bold cyan]\n")

    prompt = StructuredPrompt(
        system_instruction="You are a Senior Quantitative Risk Analyst AI. Evaluate financial risks adhering strictly to Basel III principles."
    )
    prompt.add_context(
        title="Account History",
        content="Account #8812 - Total assets: $4,250,000. 30-day average volatility: 1.8%. Liquidity ratio: 1.45.",
        source="Core_Banking_DB"
    )
    prompt.add_context(
        title="Pending Transaction",
        content="Transaction: Transfer $650,000 to Offshore Entity 'Cyprus Global Trading Ltd' at 02:45 AM UTC.",
        source="Payment_Gateway"
    )
    prompt.add_exemplar(
        user_example="Evaluate: Wire $500 to Local Grocery Supplier.",
        assistant_example='{"risk_level": "LOW", "score": 5, "decision": "AUTO_APPROVE"}'
    )
    prompt.set_user_instruction(
        "Analyze the pending transaction in light of the account history. Flag suspicious anomalies and provide risk classification."
    )
    prompt.set_output_constraint(
        OutputConstraint(
            format_type="json",
            json_schema={
                "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
                "score": "integer 0-100",
                "anomalies": ["list of strings"],
                "action": "AUTO_APPROVE | MANUAL_REVIEW | REJECT"
            },
            forbidden_phrases=["I believe", "As an AI", "Maybe"]
        )
    )

    messages = prompt.compile_messages()

    table = Table(title="Generated Chat Messages Contract")
    table.add_column("Turn", style="cyan", width=6)
    table.add_column("Role", style="bold magenta", width=12)
    table.add_column("Content Snippet", style="white")

    for idx, msg in enumerate(messages, 1):
        content_preview = msg.content if len(msg.content) < 180 else msg.content[:180] + "..."
        table.add_row(str(idx), msg.role.value.upper(), content_preview)

    console.print(table)

    console.print("\n[bold green]Compiled XML-Wrapped Format:[/bold green]")
    console.print(Panel(prompt.compile_text(PromptFormat.XML_WRAPPED), border_style="green"))


if __name__ == "__main__":
    demonstrate_prompt_structure()
