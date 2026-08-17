# RegGuard — design notes

Companion to the README. This document records *why* the system is shaped the way it is, which is the
part that does not survive in code comments.

---

## 1. The domain problem

A bank's transaction-monitoring system raises far more alerts than analysts can clear, and the large
majority are false positives. Clearing one alert means the same repetitive work every time: pull the
KYC profile, pull the transaction history, look for known typologies, check what policy obliges the
institution to do, then write it up for a compliance officer.

That work is *sequential but not fixed*. A dormant retiree receiving a large inbound transfer needs a
different investigation from a PEP wiring funds through three jurisdictions. The order of steps
depends on what the previous step found — which is exactly the shape of problem that justifies an
LLM deciding control flow rather than a hard-coded pipeline.

The last step — deciding to file a Suspicious Transaction Report, freeze an account, or exit a
customer — is a regulated decision with legal consequences. RegGuard is designed to stop there,
deliberately.

---

## 2. Why a supervisor graph, and not the alternatives

| Option | Why not |
|---|---|
| Hard-coded pipeline | Correct order differs per case; a fixed sequence wastes tool calls on irrelevant steps and cannot skip out-of-scope queries |
| Single agent with all tools | No provenance (which reasoning produced which evidence?), one giant prompt, no least privilege, and no natural place to bound cost |
| Fully autonomous agents talking to each other | Non-reproducible, hard to bound, and impossible to present to an auditor as a defensible process |
| **Supervisor + specialists (chosen)** | Fixed legal transition set, LLM chooses within it, centralised coordination, per-agent tool scoping, one place to apply guardrails and count steps |

LangGraph was chosen because it makes that topology explicit — nodes, a reviewable edge set,
reducer-based state, and first-class checkpointing with `interrupt()` for human authorisation. The
latter is what makes the human-in-the-loop requirement a framework feature rather than a bespoke
queue.

---

## 3. State design

`app/graph/state.py`. Four of the channels are reducers (`Annotated[list[...], operator.add]`):

| Channel | Kind | Purpose |
|---|---|---|
| `case_id`, `query`, `customer_id`, `lookback_days` | inputs | set once at intake |
| `findings` | **append-only** | the evidence file; each specialist appends exactly one `Finding` |
| `decisions` | **append-only** | every `RouteDecision` the LLM made, with reasoning and confidence |
| `route_history` | **append-only** | which specialists ran, in order |
| `guardrail_events` | **append-only** | every deterministic override applied to the model |
| `next_agent`, `step_count` | control | last dispatch decision, and the step budget counter |
| `overall_risk`, `report`, `approval`, `status` | outputs | the computed risk, the report, who signed off |

Append-only evidence means no node can quietly overwrite another's work, and the final state alone is
enough to answer "why did this case reach this conclusion?" — which is the question an auditor,
a model-risk reviewer and a debugging engineer all ask.

---

## 4. One investigation, step by step

```
POST /investigations
   │
   ├─ InvestigationRequest validated  (bad case_id / short query → 422, no model call)
   │
   ├─ intake                deterministic; resolves customer_id by regex
   │
   ├─ supervisor  step 1    LLM → RouteDecision(CUSTOMER, "no profile yet", 0.85)
   │                        guardrails: pass
   ├─ customer              bind_tools → get_customer_profile(C001)
   │                        → with_structured_output(Finding) → appended
   │
   ├─ supervisor  step 2    LLM → TRANSACTION
   ├─ transaction           summarise_transactions(C001, 60)
   │
   ├─ supervisor  step 3    LLM → FRAUD          (redirected if no transaction evidence)
   ├─ fraud                 score_fraud_risk → 65/100 HIGH, rules R01/R02/R04
   │                        (R01 and R08 also carry a HIGH escalation floor)
   │
   ├─ supervisor  step 4    LLM → POLICY
   ├─ policy                search_policy("structuring cash deposit threshold …")
   │
   ├─ supervisor  step 5    LLM → FINISH
   ├─ report                overall_risk computed = HIGH
   │                        with_structured_output(InvestigationReport)
   │                        case_id / customer_id / risk_level overwritten from state
   │
   └─ human_approval        interrupt(payload) → checkpoint persisted
                            HTTP 202 + thread_id

POST /investigations/{thread_id}/approval
   └─ Command(resume={approved, approver, notes}) → status=approved → END
```

Inside each specialist, `app/agents/base.py` runs a bounded loop: model call → tool calls → tool
results → model call, up to `MAX_TOOL_ITERATIONS`, then a structured-output summarisation with one
retry and a degraded-but-valid `Finding` as the final fallback. A specialist that fails is recorded
as *incomplete*, so a case is never silently under-evidenced.

---

## 5. Trust boundaries

Four kinds of untrusted input, four defences:

0. **Caller identity** → a bearer token resolved to a `Principal` with a role
   (`app/core/security.py`). Only an `approver` may authorise an outcome, and the approver recorded in
   the audit trail is the authenticated principal — never the `approver` field of the request body,
   which is ignored (and logged) when it disagrees. Authentication is off by default so the repository
   runs unconfigured; `ENVIRONMENT=production` refuses to start in that state.
1. **Caller input** → Pydantic request models. Malformed case IDs, short queries and out-of-range
   windows are rejected at the HTTP edge (422) before any spend.
2. **LLM routing output** → `RouteDecision` validation *and* `apply_guardrails`. A hallucinated route
   cannot parse; a legal-but-unwise route is overridden and the override is logged.
3. **LLM tool arguments** → every tool that takes arguments declares an explicit `args_schema`
   (`app/schemas/tool_args.py`; the three no-argument tools need none). An out-of-range
   `lookback_days` or a missing `customer_id` becomes an error `ToolMessage` the model can correct,
   not an exception that kills the graph. Arguments the model failed to serialise at all arrive in
   `invalid_tool_calls` and are answered too, so the message history stays valid for the next call.

Tool failures are converted, never raised: unknown tool, invalid arguments, domain error
(`RecordNotFoundError`) and unexpected exception all return an error message the agent can react to —
the same way a human analyst reacts to "no such customer". `tests/test_guardrails.py` covers all four.

---

## 6. Model risk: what the model is not allowed to do

A compliance system has to answer "could the model have made this number up?" with "no, structurally".

* The **risk score** is a versioned rule engine (`RULES_VERSION`), with weights, thresholds, and the
  specific transactions that triggered each rule. Deterministic, reproducible, back-testable. The
  fraud agent's prompt forbids adjusting it and its value is that it *explains* the rules.
* Some rules carry a **mandatory escalation floor** (`RuleHit.min_level`) instead of relying on their
  weight: a sanctions match scores 50 where HIGH begins at 60, and reporting it is an obligation, not
  a judgement. Floors move the level up only, and the rules that applied are reported in
  `escalated_by`. Expressing a hard obligation as additive weight is a modelling error — arithmetic
  elsewhere in the case could dilute it below the gate.
* The **headline risk level** is `max(finding.assessed_risk)`, computed in code.
* `requires_sar_filing` can be escalated by code but never de-escalated by the model.
* The **report's** `case_id`, `customer_id` and `risk_level` are overwritten from state after
  generation, so the writer cannot restate provenance incorrectly.
* **Approval necessity** is a governance rule, not a model output.
* **Approver identity** is taken from the authenticated principal. An audit trail whose reviewer name
  came from the request body records nothing an auditor can rely on.

Temperature defaults to `0.0` and applies to every call the system makes (`LLM_TEMPERATURE`,
`app/core/llm.py`): a control-flow decision should be reproducible given the same evidence, and a
report should not be creative.

---

## 6b. Time, retention and serialisation

Three properties of the runtime that are easy to get wrong quietly.

**What "now" means is configuration, not a constant.** `TIME_ANCHOR=dataset` (the default) anchors
lookback windows to the newest transaction in the shipped snapshot, so every run and every test is
deterministic whatever date it executes; `TIME_ANCHOR=now` anchors to wall-clock UTC, which is what a
live source system needs. The distinction changes results, not just plumbing: under `dataset` the
fixtures yield 12 transactions in a 30-day window, under `now` they yield 2, because the synthetic
data predates today. `reference_date()` is deliberately uncached — a cached value would freeze the
clock for the process lifetime.

**Retention is bounded.** `MemorySaver` keeps every checkpoint of every thread for ever, which is
acceptable in a script and a leak in a service: roughly one checkpoint per node, per investigation,
for the process lifetime. `BoundedMemorySaver` keeps the `MAX_RETAINED_INVESTIGATIONS` most recently
written threads and evicts the oldest, logging each eviction — so an approval that arrives after
eviction has a traceable explanation instead of looking like data loss. This is a bound, not
durability; the Postgres saver is still the answer for surviving a restart.

**Checkpointed types are registered explicitly.** State channels carry Pydantic models and enums.
LangGraph deserialises unregistered types with a warning today and will refuse to in a future release,
so with floating `langgraph>=1.2,<2.0` this was a live upgrade hazard: reading or resuming a persisted
investigation would have started failing on a minor bump. `REGGUARD_MSGPACK_TYPES` lists the six types
that may be reconstructed from a checkpoint, and the test suite proves the round trip under
`LANGGRAPH_STRICT_MSGPACK=true` — the future behaviour, exercised now. Adding a new model to state
means adding it there, and the test fails if a registered type is renamed away.

---

## 7. Failure modes considered

| Failure | Handling |
|---|---|
| Provider timeout / 5xx | `timeout` + `max_retries` on the client (`app/core/llm.py`) |
| Malformed structured output | one corrective retry, then a degraded valid object; investigation continues |
| Provider returns *no* object (refusal / plain text) | treated as a schema failure: the supervisor degrades to `FINISH` at confidence `0.0`, a specialist yields a `Finding` marked incomplete |
| Tool call with unparsable arguments | answered with an error `ToolMessage` bound to the same call id, then the loop continues |
| Model routes in a loop | visit cap → forced `FINISH`, event recorded |
| Model never finishes | step budget, checked *before* the routing call → forced `FINISH` with no wasted spend; LangGraph `recursion_limit` as a second net |
| Model asks for a nonexistent tool | error `ToolMessage` listing the available tools |
| Tool returns a huge payload | truncated at 8 000 characters before it reaches the context window |
| Unknown customer | domain error surfaced to the agent with the list of valid IDs |
| Out-of-scope query | supervisor finishes in one step; zero tool calls, zero spend |
| Missing credentials | startup `ValidationError` naming the variable, not a mid-investigation crash |
| Malformed approval payload | rejected and recorded as *not approved* |
| Approval for a case that is not paused (or a second approval) | `InvestigationNotPausedError` → `409`; the decision is never silently dropped |
| A thread reused for a second case | `ThreadAlreadyUsedError` → `409`, because append-only channels would mix the evidence |
| Approval arriving after the case was evicted from memory | reads as "not found"; the eviction itself is logged with the thread id, so the cause is recoverable from the audit trail |
| Unregistered type in a checkpoint | listed in `REGGUARD_MSGPACK_TYPES`; verified under strict serialisation so a dependency bump cannot break resume silently |
| Missing or unknown bearer token when auth is on | `401`; an analyst token on the approval endpoint is `403` |
| Production started without auth, or with the stub provider | startup `ValidationError` — the process refuses to run rather than running unsafely |

---

## 8. Extending it

* **New specialist**: add a route to `AgentRoute`, a prompt to `prompts.py`, tools to `app/tools/`, an
  entry to `SPECIALISTS`, a node function, then two lines in `app/graph/build.py` — a
  `builder.add_node(...)` and a `SPECIALIST_NODES` entry. The supervisor prompt is the only other
  place that needs to learn it exists.
* **New tool for an existing specialist**: add the function, an `args_schema`, and append it to that
  specialist's tool list. Tool order matters only to the deterministic stub, which calls the first.
* **New type in state**: add it to `REGGUARD_MSGPACK_TYPES` in `app/graph/checkpointer.py`, or resume
  will break the day LangGraph enforces its allowlist.
* **New guardrail**: add a branch to `apply_guardrails` and a test to `test_guardrails.py`. Guardrails
  are pure functions of `(decision, state)`, so they are cheap to test exhaustively.

---

## 9. Scaling and operations

* The graph is stateless per request; state lives in the checkpointer. Swap `BoundedMemorySaver` for
  the Postgres saver and the service scales horizontally, with paused cases resumable by any worker.
  Until then, retention is capped by `MAX_RETAINED_INVESTIGATIONS` (default 200) and a case older than
  that cannot be resumed — which is the practical reason to make the swap before running this for real,
  not just restart durability.
* Cost is bounded per case by at most `MAX_SUPERVISOR_STEPS` routing calls (the budget is checked
  before the call, so the limit is never exceeded and never wasted) plus, per specialist, up to
  `MAX_TOOL_ITERATIONS` tool rounds and 2 summarisation calls (one corrective retry), plus 1 report
  call. That ceiling is configuration, so it can be tuned per environment and enforced in review.
* Every log line is JSON and carries `case_id` via a `ContextVar`, so an investigation is one query in
  a log store. The natural next step is OpenTelemetry spans per node plus LangSmith traces.
* Long investigations should move to a queue: `POST /investigations` enqueues, the graph runs in a
  worker, and the client polls `GET /investigations/{thread_id}` — which already returns the full
  audit trail.

---

## 10. Known limitations

* Fixture data is synthetic and small; the retriever is lexical, not semantic.
* The rule engine encodes a handful of well-known typologies, not a bank's real typology library.
* The deterministic stub imitates a competent supervisor. It demonstrates the *mechanism* end to end;
  measuring real routing *quality* needs an offline eval set of cases with expected routes, which is
  the first thing I would build next. `tests/test_live_provider.py` checks that a real model works,
  not that it routes well — those are different questions and only the first is answered.
* Authentication is bearer tokens from configuration with two roles. It makes the approver identity
  authenticated rather than self-declared, which was the point; it is not an identity provider, and
  there is no token rotation, session revocation or rate limiting.
* No PII redaction before prompts. With real customer data that is a prerequisite, not a nicety.
* Retention is bounded but not durable: a restart loses paused investigations.
