"""
Core type definitions for Phase 5 LLM Engineering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import uuid


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool / function invocation request emitted by the LLM."""
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
        )


@dataclass
class Message:
    """Represents a single turn in a multi-turn conversation."""
    role: MessageRole
    content: str = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, MessageRole) else str(self.role),
            "content": self.content,
        }
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class TokenUsage:
    """Tracks token consumption for observability and cost accounting."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


@dataclass
class GenerationResponse:
    """Output envelope returned by LLMBackend."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", "guardrail_blocked"
    model_name: str = "mock-llm"
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    """Declarative specification for a callable agent tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required_permissions: List[str] = field(default_factory=list)
    requires_human_confirmation: bool = False

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
