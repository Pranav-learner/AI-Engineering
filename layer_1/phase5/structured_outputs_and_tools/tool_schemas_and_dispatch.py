"""
5.2 Tool Schemas, Registry, Dynamic Dispatch & Execution Sandbox.
Implements:
1. Automatic JSON Schema generation from Python functions & type hints
2. Declarative @tool decorator with RBAC permissions & Human-In-The-Loop flags
3. Tool Registry for discovery & schema export
4. Sandboxed Tool Executor with parameter coercion, validation & error envelopes
"""

from dataclasses import dataclass, field
import functools
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, get_type_hints
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from layer_1.phase5.common.types import ToolCall, ToolDefinition


# ==========================================
# Schema Reflection from Python Callables
# ==========================================

def _python_type_to_json_type(py_type: Any) -> str:
    if py_type in (int,):
        return "integer"
    elif py_type in (float,):
        return "number"
    elif py_type in (bool,):
        return "boolean"
    elif py_type in (list, List):
        return "array"
    elif py_type in (dict, Dict):
        return "object"
    return "string"


def generate_tool_schema(func: Callable) -> Dict[str, Any]:
    """Generates an OpenAI-compatible JSON Schema from a Python function signature and docstrings."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = inspect.getdoc(func) or "No description provided."

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = type_hints.get(param_name, str)
        json_type = _python_type_to_json_type(param_type)

        param_spec: Dict[str, Any] = {"type": json_type}
        
        # Check default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            param_spec["default"] = param.default

        # Heuristic description from docstrings
        param_spec["description"] = f"Parameter '{param_name}' of type {json_type}."
        properties[param_name] = param_spec

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ==========================================
# @tool Decorator & Registry
# ==========================================

@dataclass
class RegisteredTool:
    name: str
    description: str
    fn: Callable
    definition: ToolDefinition
    required_permissions: Set[str] = field(default_factory=set)
    requires_human_confirmation: bool = False


class ToolRegistry:
    """Central catalog of tools available to agents."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        requires_human_confirmation: bool = False
    ) -> Callable:
        """Decorator to register a Python callable as an agent tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or inspect.getdoc(func) or f"Tool {tool_name}"
            param_schema = generate_tool_schema(func)

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=param_schema,
                required_permissions=required_permissions or [],
                requires_human_confirmation=requires_human_confirmation,
            )

            registered = RegisteredTool(
                name=tool_name,
                description=tool_desc,
                fn=func,
                definition=definition,
                required_permissions=set(required_permissions or []),
                requires_human_confirmation=requires_human_confirmation,
            )

            self._tools[tool_name] = registered
            return func
        return decorator

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def list_definitions(self, user_permissions: Optional[Set[str]] = None) -> List[ToolDefinition]:
        """Returns tool definitions, optionally filtered by user authorization level."""
        defs = []
        for tool in self._tools.values():
            if user_permissions is not None:
                # If tool requires permissions, verify user holds them
                if tool.required_permissions and not tool.required_permissions.issubset(user_permissions):
                    continue
            defs.append(tool.definition)
        return defs


# ==========================================
# Sandboxed Tool Execution Engine
# ==========================================

@dataclass
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    is_blocked_by_auth: bool = False
    requires_human_confirmation: bool = False

    def to_message_content(self) -> str:
        if not self.success:
            return json.dumps({
                "status": "ERROR",
                "error": self.error,
                "tool": self.tool_name,
                "hint": "Verify required parameters and permissions."
            })
        if isinstance(self.output, (dict, list)):
            return json.dumps(self.output)
        return str(self.output)


class ToolExecutor:
    """Executes tool calls with strict type conversion, error catching, and authorization checks."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_tool_call(
        self,
        tool_call: ToolCall,
        user_permissions: Optional[Set[str]] = None,
        human_approved: bool = False
    ) -> ToolExecutionResult:
        tool = self.registry.get(tool_call.name)
        if not tool:
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                output=None,
                error=f"Unrecognized tool '{tool_call.name}'. Available: {list(self.registry._tools.keys())}"
            )

        # 1. Authorization Check (RBAC)
        if tool.required_permissions:
            perms = user_permissions or set()
            if not tool.required_permissions.issubset(perms):
                missing = tool.required_permissions - perms
                return ToolExecutionResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    success=False,
                    output=None,
                    error=f"Permission Denied: User lacks required permissions {missing}.",
                    is_blocked_by_auth=True
                )

        # 2. Human-in-the-loop confirmation check
        if tool.requires_human_confirmation and not human_approved:
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                output=None,
                error=f"Action '{tool_call.name}' requires explicit Human-In-The-Loop supervisor confirmation.",
                requires_human_confirmation=True
            )

        # 3. Parameter Validation and Type Coercion
        try:
            sig = inspect.signature(tool.fn)
            type_hints = get_type_hints(tool.fn)
            coerced_args = {}

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                if param_name in tool_call.arguments:
                    val = tool_call.arguments[param_name]
                    target_type = type_hints.get(param_name, None)
                    # Type coercion
                    if target_type == int and isinstance(val, str) and val.isdigit():
                        val = int(val)
                    elif target_type == float and isinstance(val, (int, str)):
                        val = float(val)
                    coerced_args[param_name] = val
                elif param.default is inspect.Parameter.empty:
                    return ToolExecutionResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        success=False,
                        output=None,
                        error=f"Missing required parameter '{param_name}' for tool '{tool_call.name}'."
                    )

            # 4. Safe Execution
            result = tool.fn(**coerced_args)
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=True,
                output=result
            )

        except Exception as ex:
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                output=None,
                error=f"Runtime Exception in tool '{tool_call.name}': {type(ex).__name__}: {str(ex)}"
            )


# ==========================================
# Sample Tool Suite
# ==========================================

global_registry = ToolRegistry()


@global_registry.register(
    name="get_account_balance",
    description="Retrieves current balance and credit line for a given account ID.",
    required_permissions=["finance:read"]
)
def get_account_balance(account_id: str, currency: str = "USD") -> Dict[str, Any]:
    """Look up balance from banking core ledger."""
    return {
        "account_id": account_id,
        "available_balance": 4250000.00,
        "currency": currency,
        "status": "ACTIVE",
        "overdraft_limit": 500000.00
    }


@global_registry.register(
    name="calculate_metric",
    description="Computes statistical variance or volatility over numerical series.",
    required_permissions=["analytics:read"]
)
def calculate_metric(metric: str, values: List[float]) -> Dict[str, Any]:
    """Perform mathematical calculation."""
    if not values:
        raise ValueError("Values list cannot be empty.")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return {
        "metric": metric,
        "count": len(values),
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "volatility_pct": round((variance ** 0.5) * 100, 2)
    }


@global_registry.register(
    name="execute_funds_transfer",
    description="Transfers money from source account to destination counterparty. HIGH IMPACT.",
    required_permissions=["finance:write"],
    requires_human_confirmation=True
)
def execute_funds_transfer(source_account: str, destination_account: str, amount: float) -> Dict[str, Any]:
    """Critical fund transfer operation requiring human confirmation."""
    return {
        "status": "SETTLED",
        "reference": "WIRE_883190",
        "source": source_account,
        "destination": destination_account,
        "amount": amount
    }


def demonstrate_tool_dispatch():
    """Demonstrates tool reflection, authorization, and dispatch."""
    console = Console()
    console.print("\n[bold cyan]═══ 5.2 TOOL REGISTRY & SANDBOXED DISPATCH ═══[/bold cyan]\n")

    executor = ToolExecutor(global_registry)

    # 1. Inspect generated JSON Schemas
    console.print("[bold yellow]1. Registered Tools & Generated Schemas:[/bold yellow]")
    table = Table(title="Agent Tool Definitions")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Permissions", style="magenta")
    table.add_column("HITL Required", justify="center")
    table.add_column("Parameters Count", justify="right")

    for td in global_registry.list_definitions():
        tool_reg = global_registry.get(td.name)
        perms = ", ".join(td.required_permissions) if td.required_permissions else "None"
        param_count = len(td.parameters.get("properties", {}))
        table.add_row(td.name, perms, "✔" if td.requires_human_confirmation else "-", str(param_count))

    console.print(table)

    # 2. Execute authorized tool call
    console.print("\n[bold yellow]2. Dispatching Authorized Tool Call (get_account_balance):[/bold yellow]")
    tc1 = ToolCall(name="get_account_balance", arguments={"account_id": "ACC_8812"})
    res1 = executor.execute_tool_call(tc1, user_permissions={"finance:read"})
    console.print(Panel(res1.to_message_content(), title="[green]Execution Result[/green]", border_style="green"))

    # 3. Execute tool call with missing permission
    console.print("\n[bold yellow]3. Attempting Tool Call with Missing Permission (execute_funds_transfer):[/bold yellow]")
    tc2 = ToolCall(name="execute_funds_transfer", arguments={"source_account": "ACC_1", "destination_account": "ACC_2", "amount": 50000.0})
    res2 = executor.execute_tool_call(tc2, user_permissions={"finance:read"})
    console.print(Panel(res2.to_message_content(), title="[red]Security Block[/red]", border_style="red"))


if __name__ == "__main__":
    demonstrate_tool_dispatch()
