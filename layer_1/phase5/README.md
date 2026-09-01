# Layer 1 Phase 5: LLM Application Engineering

> **The Core LLM Engineering Lesson:**
> 
> ```
> LLM (Probabilistic / Uncertain Component)
>   ↓
> Application Contract (Pydantic Typed Schemas)
>   ↓
> Validation (Sanitization & Self-Healing Retry Loop)
>   ↓
> Authorization (RBAC & Human-In-The-Loop Sign-Off)
>   ↓
> Tools (Dynamic Schemas & Sandboxed Execution)
>   ↓
> Policy (Deterministic Rule Enforcement)
>   ↓
> Observability (Hierarchical Spans, Latency, Token Accounting)
>   ↓
> Evaluation (Red-Team Benchmarks & Metric Scorecards)
> ```

---

## 📋 Phase 5 Completion Checklist

### 5.1 Prompt Fundamentals
- [x] **Prompt structure** (`layer_1/phase5/prompt_fundamentals/prompt_structure.py`)
- [x] **System instructions** (Role, policy, personas, negative constraints)
- [x] **User instructions** (Task isolation, task boundaries)
- [x] **Context** (Retrieved document tags, XML isolation, metadata)
- [x] **Output requirements** (Strict format contracts, forbidden tokens)
- [x] **Zero-shot** (`layer_1/phase5/prompt_fundamentals/prompt_strategies.py`)
- [x] **Few-shot** (Exemplar management and dynamic formatting)
- [x] **Chain-of-thought (CoT)** (Zero-shot CoT & Few-shot CoT traces)
- [x] **Self-consistency** (Multi-path sampling + majority voting + confidence scoring)
- [x] **ReAct** (Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Synthesis)
- [x] **Tree-of-thought (ToT)** (Branching, heuristic evaluation, beam search)
- [x] **Planning** (Plan-and-solve task decomposition and sequential step execution)
- [x] **Prompt playground implemented** (`layer_1/phase5/prompt_fundamentals/prompt_playground.py`)
- [x] **Prompt experiments completed** (`layer_1/phase5/prompt_fundamentals/prompt_experiments_and_failures.py`)
- [x] **Prompt failure modes tested** (Lost in the middle, instruction drift, sycophancy, truncation, hallucination)

### 5.2 Structured Outputs & Tools
- [x] **JSON outputs** (`layer_1/phase5/structured_outputs_and_tools/structured_outputs.py`)
- [x] **Schemas** (Pydantic V2 models & JSON Schema reflection)
- [x] **Validation** (Syntax checking, fence stripping, bracket balance repair)
- [x] **Typed outputs** (Parse directly into validated domain models)
- [x] **Error handling** (Syntactic fallback, AST repair, error telemetry)
- [x] **Retry strategies** (Self-healing feedback prompts with exponential backoff)
- [x] **Function schemas** (`layer_1/phase5/structured_outputs_and_tools/tool_schemas_and_dispatch.py`)
- [x] **Tool selection** (Dynamic intent matching and tool definition compilation)
- [x] **Arguments** (Signature reflection, type casting, parameter validation)
- [x] **Tool execution** (Sandboxed dispatch with exception catching)
- [x] **Tool results** (Standardized JSON result envelopes)
- [x] **Tool errors** (Structured error responses and hint formatting)
- [x] **Tool loops** (`layer_1/phase5/structured_outputs_and_tools/tool_loop_engine.py`)
- [x] **Tool simulator implemented** (`layer_1/phase5/structured_outputs_and_tools/tool_simulator_and_failures.py`)
- [x] **Tool failure experiments completed** (Hallucinated tools, missing args, exceptions, RBAC violation, cycles)

### 5.3 Optimization & Security
- [x] **Prompt templates** (`layer_1/phase5/optimization_and_security/prompt_optimization.py`)
- [x] **Prompt compression** (Heuristic filler stripping & entity-preserving token reduction)
- [x] **Context management** (Sliding window, pinned system prompt, conversation condensation)
- [x] **Prompt versioning** (Semantic version registry, SHA-256 integrity hashes, changelogs)
- [x] **Prompt optimization** (Metric-driven automated prompt search & evaluation)
- [x] **Prompt injection** (`layer_1/phase5/optimization_and_security/prompt_security.py`)
- [x] **Direct injection** (Instruction overrides, maintenance mode persona hijacking)
- [x] **Indirect injection** (Hidden payloads in third-party data, PDFs, emails, logs)
- [x] **Jailbreaks** (DAN archetypes, hypothetical cyberpunk framing, negation attacks)
- [x] **Defensive prompting** (Dual XML delimiter sandboxing, strict untrusted tags)
- [x] **Input validation** (Regex threat scanner, injection signature detection)
- [x] **Output validation** (Secret pattern detection, system leak filters)
- [x] **Tool authorization** (RBAC permissions, Human-In-The-Loop step-up checks)
- [x] **Secure prompt pipeline** (`layer_1/phase5/optimization_and_security/secure_pipeline.py`)
- [x] **Attack experiments completed** (`layer_1/phase5/optimization_and_security/attack_experiments.py` — 15+ automated attack vectors)

### Full Application
- [x] **Autonomous Financial Risk & Operations Decision Engine (AFRO-DE)** (`layer_1/phase5/full_decision_system.py`)

---

## 🚀 Quickstart & Execution

### 1. Interactive Master CLI Dashboard
Run the central interactive dashboard to explore any module:
```bash
python -m layer_1.phase5.main
```

### 2. Run All Pytest Tests
```bash
pytest layer_1/phase5/tests/ -v
```

### 3. Run Individual Modules Directly

```bash
# 5.1 Prompt Fundamentals
python -m layer_1.phase5.prompt_fundamentals.prompt_structure
python -m layer_1.phase5.prompt_fundamentals.prompt_strategies
python -m layer_1.phase5.prompt_fundamentals.prompt_playground
python -m layer_1.phase5.prompt_fundamentals.prompt_experiments_and_failures

# 5.2 Structured Outputs & Tools
python -m layer_1.phase5.structured_outputs_and_tools.structured_outputs
python -m layer_1.phase5.structured_outputs_and_tools.tool_schemas_and_dispatch
python -m layer_1.phase5.structured_outputs_and_tools.tool_loop_engine
python -m layer_1.phase5.structured_outputs_and_tools.tool_simulator_and_failures

# 5.3 Optimization & Security
python -m layer_1.phase5.optimization_and_security.prompt_optimization
python -m layer_1.phase5.optimization_and_security.prompt_security
python -m layer_1.phase5.optimization_and_security.secure_pipeline
python -m layer_1.phase5.optimization_and_security.attack_experiments

# Full Real-World System
python -m layer_1.phase5.full_decision_system
```

---

## 📂 Architecture & Directory Map

```
layer_1/phase5/
├── __init__.py
├── README.md                              # This comprehensive engineering guide
├── main.py                                # Master interactive terminal CLI dashboard
├── full_decision_system.py                # End-to-end Autonomous Financial Risk Decision Engine (AFRO-DE)
│
├── common/
│   ├── types.py                           # Message, ToolCall, TokenUsage, ToolDefinition dataclasses
│   ├── llm_backend.py                     # Deterministic Mock LLM + Live Provider adapters
│   └── observability.py                   # Hierarchical Spans, Tracer, Latency & Token accounting
│
├── prompt_fundamentals/
│   ├── prompt_structure.py                # System, User, Context, Output Constraint compilation
│   ├── prompt_strategies.py               # Zero/Few-shot, CoT, Self-Consistency, ToT, Planning
│   ├── prompt_playground.py               # Interactive CLI playground for experimenting with strategies
│   └── prompt_experiments_and_failures.py # Empirical test suite for 5 classic prompt failure modes
│
├── structured_outputs_and_tools/
│   ├── structured_outputs.py              # Robust JSON extraction, Pydantic schemas & self-healing retry
│   ├── tool_schemas_and_dispatch.py       # Function reflection, @tool decorator, RBAC & sandbox
│   ├── tool_loop_engine.py                # Multi-turn autonomous agent loop with cycle prevention
│   └── tool_simulator_and_failures.py     # Red-team test harness for 6 critical tool failure modes
│
├── optimization_and_security/
│   ├── prompt_optimization.py             # Jinja2 templates, token compression, sliding window & versioning
│   ├── prompt_security.py                 # Threat payloads, Input/Output guardrails, Canary tokens
│   ├── secure_pipeline.py                 # End-to-end hardened pipeline orchestration
│   └── attack_experiments.py              # Red-team benchmark (15+ attack vectors, 100% defense score)
│
└── tests/
    ├── test_prompt_fundamentals.py        # 100% test coverage for 5.1
    ├── test_structured_outputs_and_tools.py # 100% test coverage for 5.2
    └── test_optimization_and_security.py  # 100% test coverage for 5.3
```

---

## 🧠 Key Takeaways for the AI Engineer

1. **Never trust raw LLM output**:
   Wrap every LLM response in a strict Pydantic model contract. If parsing fails, use syntactic AST repair first, then trigger a targeted corrective feedback prompt with backoff.
2. **Never execute tool calls blindly**:
   Enforce Role-Based Access Control (RBAC) at the dispatcher level before Python invokes any callable. Require human supervisor sign-off for destructive or financial operations.
3. **Guardrails must be layered**:
   A single defense is not enough. Secure pipelines combine **Input Threat Scanners** + **Cryptographic Canary Tokens** + **Dual-Boundary XML Delimiters** + **Output Scrubbing** + **Deterministic Hard Policies**.
4. **Treat Prompts as Production Code**:
   Use Jinja2 templates, manage prompt versions with semantic tags (`v1.0.0`, `v1.1.0`) and SHA-256 hashes, compress token footprints, and evaluate accuracy metrics against automated test sets.
