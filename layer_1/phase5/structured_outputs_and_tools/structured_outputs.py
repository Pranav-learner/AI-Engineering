"""
5.2 Structured Outputs, Schema Validation & Self-Healing Retry Strategies.
Implements:
1. JSON Output Extraction (Markdown fence removal, regex boundary locator, syntax repair)
2. Schema Definition (Pydantic models and JSON Schema)
3. Strict Type Validation & Detailed Error Telemetry
4. Self-Healing Feedback Loop (Targeted repair prompts with backoff)
"""

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from layer_1.phase5.common.types import Message, MessageRole, GenerationResponse
from layer_1.phase5.common.llm_backend import LLMBackend, get_llm_backend
from layer_1.phase5.common.observability import Tracer


T = TypeVar("T", bound=BaseModel)


# ==========================================
# Domain Schemas (Pydantic Models)
# ==========================================

class FinancialRiskAssessment(BaseModel):
    """Structured contract for financial risk scoring."""
    transaction_id: str = Field(description="Unique transaction reference code, e.g. TX_1092")
    risk_level: str = Field(description="Categorical risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    risk_score: float = Field(ge=0.0, le=100.0, description="Numerical risk score between 0 and 100")
    flagged_factors: List[str] = Field(default_factory=list, description="Specific risk indicators identified")
    approval_required: bool = Field(description="Whether a human supervisor must authorize the transfer")
    recommended_action: str = Field(description="Recommended policy action: APPROVE, REJECT, STEP_UP_AUTH")


class UserProfile(BaseModel):
    """Structured contract for user profiling."""
    user_id: str
    username: str
    email: str
    is_active: bool
    roles: List[str]
    credit_score: int = Field(ge=300, le=850)


# ==========================================
# Robust JSON Extractor & Heuristic Repairer
# ==========================================

class JSONExtractor:
    """Extracts and repairs JSON payloads from conversational or noisy LLM output."""

    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """Extracts JSON substring from markdown code blocks or curly bracket boundaries."""
        text = raw_text.strip()

        # 1. Check for markdown code fence ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        # 2. Check for outermost curly brackets { ... }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

        # 3. Check for outermost square brackets [ ... ]
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return text[start_arr : end_arr + 1].strip()

        return text

    @staticmethod
    def attempt_syntactic_repair(json_str: str) -> str:
        """Heuristically repairs common LLM syntax errors (trailing commas, unclosed quotes)."""
        repaired = json_str.strip()

        # Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Fix unclosed curly brace
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        # Fix unclosed square bracket
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)

        return repaired


# ==========================================
# Typed Parser with Self-Healing Retry Loop
# ==========================================

@dataclass
class ParseResult(Generic[T]):
    parsed_object: Optional[T]
    raw_response: str
    attempts: int
    validation_errors: List[str]
    success: bool


class StructuredOutputEngine:
    """
    Enforces typed application contracts on top of probabilistic LLMs.
    Features:
    - Pydantic schema compilation into prompt instructions.
    - Robust multi-stage JSON extraction.
    - Automatic self-healing repair feedback loop with exponential backoff.
    """

    def __init__(self, backend: LLMBackend, max_retries: int = 3, initial_delay_sec: float = 0.1):
        self.backend = backend
        self.max_retries = max_retries
        self.initial_delay_sec = initial_delay_sec

    def parse_structured(
        self,
        schema: Type[T],
        prompt_messages: List[Message],
        tracer: Optional[Tracer] = None
    ) -> ParseResult[T]:
        """
        Executes prompt and returns a validated Pydantic model.
        If validation fails, constructs a precise corrective feedback prompt and retries.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_contract = (
            f"You MUST output valid JSON strictly conforming to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Output ONLY the JSON object. No explanations, no markdown intro."
        )

        conversation = list(prompt_messages)
        # Ensure system contract is present
        if conversation and conversation[0].role == MessageRole.SYSTEM:
            conversation[0].content += f"\n\n{system_contract}"
        else:
            conversation.insert(0, Message.system(system_contract))

        validation_errors: List[str] = []
        last_raw_content = ""

        for attempt in range(1, self.max_retries + 1):
            span = None
            if tracer:
                span = tracer.start_span(
                    name=f"StructuredOutput.Attempt_{attempt}",
                    inputs={"model": schema.__name__, "attempt": attempt}
                )

            resp = self.backend.generate(conversation, tracer=tracer)
            last_raw_content = resp.content

            # Stage 1: Extract candidate JSON
            candidate_json = JSONExtractor.extract_json_string(resp.content)

            # Stage 2: Attempt standard JSON parse
            try:
                data = json.loads(candidate_json)
            except json.JSONDecodeError as jde:
                # Stage 3: Syntactic repair attempt
                repaired = JSONExtractor.attempt_syntactic_repair(candidate_json)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    err_msg = f"JSON Syntax Error on attempt {attempt}: {str(jde)}"
                    validation_errors.append(err_msg)
                    if tracer and span:
                        tracer.end_span(span, status="ERROR", error=err_msg)
                    
                    # Formulate corrective feedback
                    conversation.append(Message.assistant(resp.content))
                    conversation.append(Message.user(
                        f"CORRECTIVE ACTION REQUIRED:\n"
                        f"Your output was not valid JSON ({str(jde)}).\n"
                        f"Please fix the syntax and output ONLY valid JSON matching the schema."
                    ))
                    time.sleep(self.initial_delay_sec * (2 ** (attempt - 1)))
                    continue

            # Stage 4: Validate against Pydantic model
            try:
                validated_model = schema.model_validate(data)
                if tracer and span:
                    tracer.end_span(span, outputs={"status": "VALIDATED", "model": str(validated_model)})
                return ParseResult(
                    parsed_object=validated_model,
                    raw_response=last_raw_content,
                    attempts=attempt,
                    validation_errors=validation_errors,
                    success=True
                )
            except ValidationError as ve:
                err_msg = f"Schema Validation Error on attempt {attempt}:\n{str(ve)}"
                validation_errors.append(err_msg)
                if tracer and span:
                    tracer.end_span(span, status="ERROR", error=err_msg)

                # Formulate corrective schema feedback prompt
                conversation.append(Message.assistant(resp.content))
                conversation.append(Message.user(
                    f"SCHEMA VALIDATION FAILURE:\n"
                    f"The JSON provided did not satisfy the required schema constraints:\n{str(ve)}\n\n"
                    f"Please re-generate the JSON with corrected fields."
                ))
                time.sleep(self.initial_delay_sec * (2 ** (attempt - 1)))

        return ParseResult(
            parsed_object=None,
            raw_response=last_raw_content,
            attempts=self.max_retries,
            validation_errors=validation_errors,
            success=False
        )


def demonstrate_structured_outputs():
    """Demonstrates typed JSON output extraction and validation."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.2 STRUCTURED OUTPUTS & SELF-HEALING REPAIRS ═══[/bold cyan]\n")

    backend = get_llm_backend("mock")
    engine = StructuredOutputEngine(backend=backend, max_retries=3)
    tracer = Tracer("StructuredOutputs-Trace")

    prompt = [
        Message.system("You are an autonomous compliance evaluator."),
        Message.user("Analyze transaction TX_9941: $85,000 international transfer to offshore shell company.")
    ]

    console.print("[bold yellow]1. Parsing Structured Pydantic Model (FinancialRiskAssessment):[/bold yellow]")
    result = engine.parse_structured(FinancialRiskAssessment, prompt, tracer=tracer)

    if result.success and result.parsed_object:
        obj = result.parsed_object
        console.print(f"[bold green]✔ Successfully parsed model in {result.attempts} attempt(s)![/bold green]")
        
        table = Table(title="Validated FinancialRiskAssessment Object")
        table.add_column("Field", style="cyan", width=22)
        table.add_column("Value", style="bold white")
        table.add_row("Transaction ID", obj.transaction_id)
        table.add_row("Risk Level", f"[bold red]{obj.risk_level}[/bold red]")
        table.add_row("Risk Score", f"{obj.risk_score:.1f} / 100")
        table.add_row("Flagged Factors", ", ".join(obj.flagged_factors))
        table.add_row("Approval Required", str(obj.approval_required))
        table.add_row("Recommended Action", obj.recommended_action)
        console.print(table)
    else:
        console.print(f"[bold red]❌ Failed after {result.attempts} attempts.[/bold red]")

    # 2. Simulate Self-Healing from Corrupted JSON
    console.print("\n[bold yellow]2. Demonstrating Self-Healing Repair on Corrupted JSON Input:[/bold yellow]")
    corrupt_backend = get_llm_backend("mock", fuzz_mode="corrupt_json")
    fuzz_engine = StructuredOutputEngine(backend=corrupt_backend, max_retries=2)
    fuzz_result = fuzz_engine.parse_structured(FinancialRiskAssessment, prompt)
    console.print(f"Self-healing caught errors: [yellow]{len(fuzz_result.validation_errors)} error(s) logged and recovered/handled.[/yellow]")


if __name__ == "__main__":
    demonstrate_structured_outputs()
