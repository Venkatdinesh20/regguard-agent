# RegGuard

**A multi-agent financial-crime investigation system whose control flow is decided by an LLM.**

Python 3.11+ · LangGraph · LangChain · FastAPI · Pydantic · 154 tests · 96% coverage · runs with no API key

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
| Production ready | Typed config with fail-fast startup validation, bearer-token auth, structured audit logging, guardrails, error handling, human-in-the-loop, bounded retention, FastAPI service, 154 tests, mypy, ruff, Docker, CI |

---

## Quickstart — no API key required

```bash
git clone <your-repo-url> regguard-agent
cd regguard-agent

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev]"            # resolves the declared ranges
# or, for the exact versions this was verified against:
# pip install -r requirements.lock && pip install -e . --no-deps

python -m app.main                 # five demo investigations, end to end
pytest                             # 154 tests
```

The default provider is a deterministic in-process model (`LLM_PROVIDER=stub`,
`app/core/stub_llm.py`) that implements the same `bind_tools` / `with_structured_output` interface as
OpenAI and Anthropic. It exists so the graph, the guardrails, the tools, the API and the
human-in-the-loop pause are all runnable and testable **without credentials, network access or
spend** — including in CI. It is a test double, never business logic.

A captured run of all five demo cases — HIGH with a human pause, MEDIUM, LOW, and an out-of-scope
query that finishes in one step without spending a tool call — is in
[`docs/sample-run.txt`](docs/sample-run.txt).

Python 3.11 is the supported floor (`requires-python`, ruff `target-version`); the container image is
3.12 and CI runs both. `requirements.lock` pins every transitive version — the declared ranges in
`pyproject.toml` are deliberately looser so upstream drift is visible, and CI installs from the ranges
for exactly that reason.

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

### Verification status

Being precise about this, because "it passes 154 tests" and "it works against a real model" are
different claims:

| | Executed |
|---|---|
| Graph, routing, guardrails, tools, rule engine, API, human-in-the-loop pause and resume, auth | ✅ every commit, via the deterministic provider |
| Checkpoint round-trip under LangGraph's future strict serialisation | ✅ `tests/test_checkpointer.py` sets `LANGGRAPH_STRICT_MSGPACK=true` |
| Real provider tool calling and structured output | ⚠️ **not executed here** — needs a key |

The third row is covered by an opt-in suite rather than a promise:

```bash
LLM_PROVIDER=openai LLM_MODEL=gpt-4.1-mini OPENAI_API_KEY=sk-... make test-live
```

It asserts what only a real model can prove: that routing decisions parse (no `confidence == 0.0`
degraded fallbacks), that tool calling produced real evidence, that the deterministic score reaches
the report unaltered, and that a HIGH-risk case still stops for a human. It is deselected from the
default run (`-m 'not live'`) so the suite stays free and offline.

### As a service

```bash
uvicorn app.api:app --reload                          # http://localhost:8000/docs
docker build -t regguard . && docker run -p 8000:8000 regguard
# once you have a .env with real credentials:
# docker run -p 8000:8000 --env-file .env regguard
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
| Step budget (`MAX_SUPERVISOR_STEPS`) | An investigation may not run — or bill — forever | Finishes **without spending another routing call**, records `STEP_BUDGET_EXHAUSTED` |
| Evidence ordering | A risk score with no activity data is not defensible to a regulator | `FRAUD` before `TRANSACTION` is redirected, records `ORDERING_VIOLATION` |
| Loop protection (`MAX_VISITS_PER_AGENT`) | Re-running an expensive specialist is a classic agent failure | Forces `FINISH`, records `LOOP_DETECTED` |
| Tool iteration cap | One specialist must not spin on its tools | Loop exits, warning logged |
| Unknown route fallback | Defence in depth if a state channel is corrupted | Routes to `FINISH` |
| Schema failure fallback | A malformed decision must not dispatch blindly | Stops with confidence `0.0` and a recorded reason |
| Provenance assertion | A specialist may not claim to be another specialist | The runner re-asserts `finding.agent` (`app/agents/base.py`) |
| Unanswered tool calls | A provider rejects an assistant tool call with no tool result, and the specialist would gather nothing | Arguments the model failed to serialise get an error `ToolMessage` too (`invalid_tool_call_message`) |
| Mandatory escalation | A sanctions match scores 50 of the 60 needed for HIGH — an obligation must not be diluted by arithmetic | `R01` and `R08` carry a `min_level` floor of HIGH, reported in `escalated_by` |
| One thread, one case | Append-only channels mean thread reuse would report case B on case A's evidence | `ThreadAlreadyUsedError` → HTTP 409 |
| Approval integrity | A reviewer must never be told a decision was recorded when it was discarded | Approving a case that is not paused → `409 Conflict`, not a silent 200 |

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
| Which specialist a finding came from | Framework code in `app/agents/base.py` | Provenance is asserted, not claimed |
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
  "pending_approval": {"type": "approval_required", "requires_sar_filing": true,
                       "question": "Authorise this recommendation? Respond with approved, approver and optional notes.",
                       "summary": "...", "recommended_actions": ["..."]}
}
```

### Authentication and who signs off

Off by default so the repo runs out of the box. Switched on, an approval is attributed to the
**authenticated principal** and any `approver` in the request body is ignored and logged as an
override attempt — the audit trail cannot be forged from the payload.

```env
AUTH_ENABLED=true
API_TOKENS=tok-analyst:a.analyst@bank.example:analyst,tok-approve:c.officer@bank.example:approver
```

```bash
curl -H "Authorization: Bearer tok-analyst" ...    # open and read cases
curl -H "Authorization: Bearer tok-approve" ...    # additionally authorise outcomes
```

`analyst` opening a case gets `202`; the same token on the approval endpoint gets **`403`**. A missing
or unknown token gets **`401`**. `ENVIRONMENT=production` **refuses to start** unless `AUTH_ENABLED`
is true and the provider is not the stub (`app/core/config.py`) — the one governance rule that cannot
be left as documentation.

Tokens in configuration are not how a bank does this; the seam for OIDC is
`app/core/security.py::principal_for_token`, and the endpoints do not change.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness and effective configuration (provider, auth posture, time anchor) — public |
| `POST` | `/investigations` | run an investigation → `200` complete, `202` awaiting authorisation |
| `GET` | `/investigations/{thread_id}` | current state and full audit trail |
| `POST` | `/investigations/{thread_id}/approval` | record a human decision and resume — `409` if the case is not awaiting one |

OpenAPI docs at `/docs`.

---

## Testing

```bash
pytest                                          # 154 tests, ~7s, no network
pytest --cov=app --cov-report=term-missing      # 96%
pytest -m live                                  # opt-in: real provider (see above)
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
| `test_auth.py` | 401/403, token parsing failures, and that a body `approver` cannot impersonate an authenticated one |
| `test_checkpointer.py` | retention eviction, and a state round-trip under `LANGGRAPH_STRICT_MSGPACK=true` |
| `test_configuration_surface.py` | both time anchors, and that `.env.example` parses to the values it appears to declare |
| `test_live_provider.py` | the opt-in real-model checks (skipped without a key) |
| `test_regressions.py` | one test per defect found in review: score monotonicity, mandatory escalation floors, a provider returning no structured object, unparsable tool arguments, double approval, thread reuse, and that the step budget costs no wasted model call |

CI (`.github/workflows/ci.yml`) runs three jobs: **quality** on Python 3.11 and 3.12 (lint, format,
types, tests with an 85% coverage floor, plus an end-to-end agent run); **locked**, which installs
`requirements.lock` and re-runs the suite so the exact verified versions stay provably working; and
**docker**, which builds the image and waits for the container's `/health` to answer.

---

## Layout

```
app/
  core/       config (pydantic-settings) · llm factory · stub model · JSON logging ·
              exceptions · security (bearer tokens, principals, roles)
  schemas/    investigation · routing · findings · tool arguments   ← every contract
  tools/      customer · transactions · fraud rule engine · policy retrieval · repository
  agents/     supervisor · specialists · reporter · base tool loop · prompts · context
  graph/      state channels · graph assembly · bounded checkpointer + serde allowlist
  data/       synthetic customers, transactions, policy corpus
  api.py      FastAPI service      main.py  CLI      service.py  application boundary
tests/        154 tests across 15 files
requirements.lock   exact verified versions
```

`app/agents/prompts.py` holds every prompt in one file: prompts are behaviour, so they are reviewed
and diffed like code, not scattered inline.

---

## Production hardening — the honest list

Everything below is a deliberate seam, not an oversight. Each swap is local.

| Today | Production | Where |
|---|---|---|
| Bounded in-memory checkpointer (retention capped, LRW eviction logged) | `langgraph-checkpoint-postgres` so paused cases survive restarts and resume on any worker | `app/graph/checkpointer.py` |
| JSON fixtures | Core-banking / case-management adapters | `app/tools/repository.py` |
| Lexical policy retrieval | Vector store + embeddings, same tool contract | `app/tools/policy.py` |
| Rule engine | Governed ML model behind the same interface, rules retained as explanations | `app/tools/fraud.py` |
| Bearer tokens from configuration, analyst/approver roles, approver taken from the token | OIDC-verified principal and roles, immutable approval log | `app/core/security.py::principal_for_token` |
| Structured logs | OpenTelemetry traces + LangSmith, per-case token and cost budgets | `app/core/logging.py` |
| — | PII redaction before prompts, prompt versioning, and an offline eval set for routing quality | new module |

Fixture data is entirely synthetic. Policy documents paraphrase publicly known AML concepts
(threshold reporting, structuring, EDD, PEP, layering, no tipping off) for demonstration and are not
legal advice.

---

## License

MIT — see `LICENSE`.
