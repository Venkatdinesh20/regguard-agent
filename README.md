# RegGuard

**A multi-agent financial-crime investigation system whose control flow is decided by an LLM.**

Python 3.11+ · LangGraph · LangChain · FastAPI · Pydantic · 105 tests · 95% coverage · runs with no API key

---

RegGuard automates the first hour of an anti-money-laundering (AML) alert investigation. An LLM
**supervisor** reads the case, decides which specialist agent should act next, and keeps deciding
until the evidence supports a defensible conclusion. Four specialist agents gather that evidence with
their own tools. A deterministic rule engine scores the risk. A human authorises anything adverse.

```
Analyst: "Investigate unusual cash deposit activity for customer C001"

  supervisor → CUSTOMER      "no profile yet; establish the baseline first"
  supervisor → TRANSACTION   "need activity data before it can be scored"
  supervisor → FRAUD         "activity shows sub-threshold cash deposits; score it"
  supervisor → POLICY        "structuring indicators found; retrieve the obligations"
  supervisor → FINISH        "evidence is sufficient to write the report"

  → HIGH risk (65/100, R01_STRUCTURING) → PAUSED for human authorisation
```

Nothing in the code says "customer, then transactions, then fraud, then policy". That order is the
model's decision on this case, recorded with its reasoning, and it changes with the case.

---

## Assignment mapping

| Requirement | Where it lives |
|---|---|
| An agent that automates a problem in a chosen field | AML / financial-crime alert triage — `app/agents/`, `app/tools/` |
| **Control flow decided by the LLM** | `app/agents/supervisor.py` emits a validated `RouteDecision`; `app/graph/build.py` dispatches on it via conditional edges |
| Multi-agent architecture | LangGraph supervisor pattern: 1 supervisor + 4 tool-owning specialists + reporter — `app/agents/specialists.py` |
| Production ready | Typed config, structured audit logging, guardrails, error handling, human-in-the-loop, FastAPI service, 105 tests, mypy, ruff, Docker, CI |

---

## Quickstart — no API key required

```bash
git clone <your-repo-url> regguard-agent
cd regguard-agent

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev]"

python -m app.main                 # five demo investigations, end to end
pytest                             # 105 tests
```

The default provider is a deterministic in-process model (`LLM_PROVIDER=stub`,
`app/core/stub_llm.py`) that implements the same `bind_tools` / `with_structured_output` interface as
OpenAI and Anthropic. It exists so the graph, the guardrails, the tools, the API and the
human-in-the-loop pause are all runnable and testable **without credentials, network access or
spend** — including in CI. It is a test double, never business logic.

A captured run of all five demo cases — HIGH with a human pause, MEDIUM, LOW, and an out-of-scope
query that finishes in one step without spending a tool call — is in
[`docs/sample-run.txt`](docs/sample-run.txt).

### Running against a real LLM

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...
```

or

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Then `python -m app.main` again. No code changes — provider is configuration
(`app/core/llm.py` is the only place a client is constructed, and
`tests/test_config_and_llm.py` proves all three paths build the same interface).

### As a service

```bash
uvicorn app.api:app --reload      # http://localhost:8000/docs
docker build -t regguard . && docker run -p 8000:8000 --env-file .env regguard
```

---

## How the LLM decides control flow

**1. The decision is a schema, not a string.**

```python
class AgentRoute(StrEnum):
    CUSTOMER = "CUSTOMER"
    TRANSACTION = "TRANSACTION"
    FRAUD = "FRAUD"
    POLICY = "POLICY"
    FINISH = "FINISH"


class RouteDecision(BaseModel):
    next_agent: AgentRoute  # an invented route is a ValidationError, not a mis-dispatch
    reasoning: str = Field(min_length=10, max_length=500)  # written to the audit trail
    confidence: float = Field(ge=0.0, le=1.0)
```

**2. The graph dispatches on it.**

```python
builder.add_conditional_edges(
    "supervisor",
    route_from_state,
    {
        "CUSTOMER": "customer",
        "TRANSACTION": "transaction",
        "FRAUD": "fraud",
        "POLICY": "policy",
        "FINISH": "report",
    },
)
for node in ("customer", "transaction", "fraud", "policy"):
    builder.add_edge(node, "supervisor")  # every specialist reports back
```

The *set* of legal transitions is fixed and reviewable. The *choice* among them, on every turn, is
the model's. That is the difference between a state machine with an LLM inside it and an unbounded
agent loop.

**3. Every decision is auditable.** The API response and the JSON logs contain each routing
decision, its reasoning and its confidence, in order:

```json
{"level":"INFO","case_id":"CASE-1001","message":"supervisor.decision","step":3,
 "model_choice":"FRAUD","dispatched":"FRAUD","confidence":0.85,
 "reasoning":"Transaction evidence shows sub-threshold cash deposits; score it.",
 "guardrail_events":[]}
```

### When the model decides badly — guardrails

LLM-decided control flow is not the same as *unbounded* control flow. The supervisor trusts the
model to choose and then verifies the choice (`app/agents/supervisor.py::apply_guardrails`). Every
override is recorded, never silent.

| Guardrail | Why | Effect |
|---|---|---|
| Step budget (`MAX_SUPERVISOR_STEPS`) | An investigation may not run — or bill — forever | Forces `FINISH`, records `STEP_BUDGET_EXHAUSTED` |
| Evidence ordering | A risk score with no activity data is not defensible to a regulator | `FRAUD` before `TRANSACTION` is redirected, records `ORDERING_VIOLATION` |
| Loop protection (`MAX_VISITS_PER_AGENT`) | Re-running an expensive specialist is a classic agent failure | Forces `FINISH`, records `LOOP_DETECTED` |
| Tool iteration cap | One specialist must not spin on its tools | Loop exits, warning logged |
| Unknown route fallback | Defence in depth if a state channel is corrupted | Routes to `FINISH` |
| Schema failure fallback | A malformed decision must not dispatch blindly | Stops with confidence `0.0` and a recorded reason |
| Provenance assertion | A specialist may not claim to be another specialist | Graph overwrites `finding.agent` |

---

## Architecture

```
                     START
                       │
                       ▼
                    intake  (deterministic: resolve the customer, no model call)
                       │
                       ▼
   ┌──────────────► supervisor ──────────────┐  LLM decides, guardrails verify
   │                   │                     │
   │   ┌───────────────┼───────────┬─────────┴──────┐
   │   ▼               ▼           ▼                ▼
   │ customer      transaction    fraud           policy      ← specialists, own tools only
   │   │               │           │                │
   └───┴───────────────┴───────────┴────────────────┘  findings appended to state
                       │
                       ▼  (FINISH)
                    report      structured InvestigationReport
                       │
                risk == HIGH ?
              ┌────────┴────────┐
              ▼                 ▼
       human_approval          END        ← interrupt(): nothing adverse without a human
              │
              ▼
             END
```

| Agent | Owns | Answers |
|---|---|---|
| **supervisor** | no tools | "who acts next, and why?" |
| **customer** | `get_customer_profile`, `list_known_customers` | tenure, residence, KYC status, PEP, sanctions, spend baseline |
| **transaction** | `summarise_transactions`, `list_transactions` | volumes, cash deposits, cross-border flows, baseline ratios |
| **fraud** | `score_fraud_risk`, `describe_risk_rules` | the deterministic score and which rules fired |
| **policy** | `search_policy`, `list_policies` | obligations, deadlines, prohibitions, with citable IDs |
| **reporter** | no tools | the analyst-facing report |

Specialists never call each other and never write to another channel of state. All coordination goes
through the supervisor, so the whole run reconstructs from the audit trail. Tool ownership is
least privilege made reviewable: the customer agent has no path to the risk engine.

---

## What is deliberately *not* the LLM's decision

This is the part that matters in a regulated domain.

| Decision | Who makes it | Why |
|---|---|---|
| The risk score | Deterministic rule engine, `app/tools/fraud.py` (10 versioned rules, weights, evidence per rule) | A model-generated number cannot be reproduced, back-tested or defended to an auditor. Same evidence → same score, every time |
| The case's headline risk level | Computed as the maximum risk any specialist evidenced | Reproducible from the evidence, not from the writer's impression |
| Whether a human must approve | Governance rule in `needs_human_approval` | A model may not opt out of oversight |
| Which specialist a finding came from | The graph | Provenance is asserted, not claimed |
| Resolving the customer identifier | Regex in `intake_node` | A parsing problem, not a reasoning problem — cheaper, and it cannot hallucinate |

The LLM decides **orchestration and explanation**. Deterministic code decides **measurement and
authority**. The rule engine sits behind a tool interface precisely so a governed ML model
(e.g. gradient-boosted classifier) can replace it later without touching the agents.

---

## Human-in-the-loop

A HIGH-risk case produces a report and then **stops**, via LangGraph's `interrupt()` with a
checkpointer. No regulatory report is filed and no account is touched.

```bash
# 1. open the case — returns 202 Accepted with a thread_id
curl -sX POST localhost:8000/investigations -H 'content-type: application/json' -d '{
  "case_id":"CASE-1001",
  "query":"Investigate unusual cash deposit activity for customer C001",
  "customer_id":"C001","lookback_days":60}'

# 2. a qualified human authorises it — the graph resumes from exactly where it paused
curl -sX POST localhost:8000/investigations/<thread_id>/approval \
  -H 'content-type: application/json' \
  -d '{"approved":true,"approver":"compliance.officer@bank.example","notes":"Structuring confirmed."}'
```

```json
{
  "status": "awaiting_approval",
  "risk_level": "HIGH",
  "route_history": ["CUSTOMER", "TRANSACTION", "FRAUD", "POLICY"],
  "decisions": [{"next_agent": "CUSTOMER", "reasoning": "...", "confidence": 0.85}, "..."],
  "guardrail_events": [],
  "report": {"requires_sar_filing": true, "recommended_actions": ["..."]},
  "pending_approval": {"question": "Authorise this recommendation?", "requires_sar_filing": true}
}
```

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness and effective configuration |
| `POST` | `/investigations` | run an investigation → `200` complete, `202` awaiting authorisation |
| `GET` | `/investigations/{thread_id}` | current state and full audit trail |
| `POST` | `/investigations/{thread_id}/approval` | record a human decision and resume |

OpenAPI docs at `/docs`.

---

## Testing

```bash
pytest                                          # 105 tests, ~7s, no network
pytest --cov=app --cov-report=term-missing      # 95%
ruff check . && ruff format --check . && mypy app tests
```

| Suite | Covers |
|---|---|
| `test_schemas.py` | every boundary rejects bad data — including a hallucinated route name |
| `test_tools.py` | expected risk outcome **per fixture customer**, rule evidence, determinism, LLM-supplied argument validation |
| `test_guardrails.py` | each guardrail fires; unknown tools, bad arguments and domain errors become messages the agent can react to, not crashes |
| `test_stub_llm.py` | the deterministic provider's contract with the prompt format |
| `test_graph.py` | full runs: HIGH pauses, approval and rejection resume, LOW completes, out-of-scope finishes in one step, step budget terminates a run |
| `test_api.py` | HTTP contract, 422s, the 202 → approve → 200 lifecycle |
| `test_config_and_llm.py` | startup validation, and that all three providers construct the interface the agents use |
| `test_cli_and_logging.py` | CLI output, JSON audit records, correlation IDs |

CI (`.github/workflows/ci.yml`) runs lint, format, types, tests with an 85% coverage floor, an
end-to-end agent run, and a Docker build with a live health check — on Python 3.11 and 3.12.

---

## Layout

```
app/
  core/       config (pydantic-settings) · llm factory · stub model · JSON logging · exceptions
  schemas/    investigation · routing · findings · tool arguments   ← every contract
  tools/      customer · transactions · fraud rule engine · policy retrieval · repository
  agents/     supervisor · specialists · reporter · base tool loop · prompts · context
  graph/      state channels · graph assembly
  data/       synthetic customers, transactions, policy corpus
  api.py      FastAPI service      main.py  CLI      service.py  application boundary
tests/        105 tests
```

`app/agents/prompts.py` holds every prompt in one file: prompts are behaviour, so they are reviewed
and diffed like code, not scattered inline.

---

## Production hardening — the honest list

Everything below is a deliberate seam, not an oversight. Each swap is local.

| Today | Production | Where |
|---|---|---|
| `MemorySaver` checkpointer | `langgraph-checkpoint-postgres` so paused cases survive restarts and resume on any worker | `app/graph/build.py::get_graph` |
| JSON fixtures | Core-banking / case-management adapters | `app/tools/repository.py` |
| Lexical policy retrieval | Vector store + embeddings, same tool contract | `app/tools/policy.py` |
| Rule engine | Governed ML model behind the same interface, rules retained as explanations | `app/tools/fraud.py` |
| Approver in the request body | Authenticated principal (OIDC), role check, immutable approval log | `app/api.py` |
| Structured logs | OpenTelemetry traces + LangSmith, per-case token and cost budgets | `app/core/logging.py` |
| — | PII redaction before prompts, prompt versioning, and an offline eval set for routing quality | new module |

Fixture data is entirely synthetic. Policy documents paraphrase publicly known AML concepts
(threshold reporting, structuring, EDD, PEP, layering, no tipping off) for demonstration and are not
legal advice.

---

## License

MIT — see `LICENSE`.
