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

`app/graph/state.py`. Three of the channels are reducers (`Annotated[list[...], operator.add]`):

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
   ├─ supervisor  step 1    LLM → RouteDecision(CUSTOMER, "no profile yet", 0.9)
   │                        guardrails: pass
   ├─ customer              bind_tools → get_customer_profile(C001)
   │                        → with_structured_output(Finding) → appended
   │
   ├─ supervisor  step 2    LLM → TRANSACTION
   ├─ transaction           summarise_transactions(C001, 60)
   │
   ├─ supervisor  step 3    LLM → FRAUD          (redirected if no transaction evidence)
   ├─ fraud                 score_fraud_risk → 65/100 HIGH, rules R01/R02/R04
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

Three kinds of untrusted input, three defences:

1. **Caller input** → Pydantic request models. Malformed case IDs, short queries and out-of-range
   windows are rejected at the HTTP edge (422) before any spend.
2. **LLM routing output** → `RouteDecision` validation *and* `apply_guardrails`. A hallucinated route
   cannot parse; a legal-but-unwise route is overridden and the override is logged.
3. **LLM tool arguments** → every tool declares an explicit `args_schema` (`app/schemas/tool_args.py`).
   An out-of-range `lookback_days` or a missing `customer_id` becomes an error `ToolMessage` the model
   can correct, not an exception that kills the graph.

Tool failures are converted, never raised: unknown tool, invalid arguments, domain error
(`RecordNotFoundError`) and unexpected exception all return an error message the agent can react to —
the same way a human analyst reacts to "no such customer". `tests/test_guardrails.py` covers all four.

---

## 6. Model risk: what the model is not allowed to do

A compliance system has to answer "could the model have made this number up?" with "no, structurally".

* The **risk score** is a versioned rule engine (`RULES_VERSION`), with weights, thresholds, and the
  specific transactions that triggered each rule. Deterministic, reproducible, back-testable. The
  fraud agent's prompt forbids adjusting it and its value is that it *explains* the rules.
* The **headline risk level** is `max(finding.assessed_risk)`, computed in code.
* `requires_sar_filing` can be escalated by code but never de-escalated by the model.
* The **report's** `case_id`, `customer_id` and `risk_level` are overwritten from state after
  generation, so the writer cannot restate provenance incorrectly.
* **Approval necessity** is a governance rule, not a model output.

Temperature is pinned to `0.0` for routing: a control-flow decision should be reproducible given the
same evidence.

---

## 7. Failure modes considered

| Failure | Handling |
|---|---|
| Provider timeout / 5xx | `timeout` + `max_retries` on the client (`app/core/llm.py`) |
| Malformed structured output | one corrective retry, then a degraded valid object; investigation continues |
| Model routes in a loop | visit cap → forced `FINISH`, event recorded |
| Model never finishes | step budget → forced `FINISH`; LangGraph `recursion_limit` as a second net |
| Model asks for a nonexistent tool | error `ToolMessage` listing the available tools |
| Tool returns a huge payload | truncated at 8 000 characters before it reaches the context window |
| Unknown customer | domain error surfaced to the agent with the list of valid IDs |
| Out-of-scope query | supervisor finishes in one step; zero tool calls, zero spend |
| Missing credentials | startup `ValidationError` naming the variable, not a mid-investigation crash |
| Malformed approval payload | rejected and recorded as *not approved* |

---

## 8. Extending it

* **New specialist**: add a route to `AgentRoute`, a prompt to `prompts.py`, tools to `app/tools/`, an
  entry to `SPECIALISTS`, a node function, and one line in `SPECIALIST_NODES`. The supervisor prompt
  is the only other place that needs to learn it exists.
* **New tool for an existing specialist**: add the function, an `args_schema`, and append it to that
  specialist's tool list. Tool order matters only to the deterministic stub, which calls the first.
* **New guardrail**: add a branch to `apply_guardrails` and a test to `test_guardrails.py`. Guardrails
  are pure functions of `(decision, state)`, so they are cheap to test exhaustively.

---

## 9. Scaling and operations

* The graph is stateless per request; state lives in the checkpointer. Swap `MemorySaver` for the
  Postgres saver and the service scales horizontally, with paused cases resumable by any worker.
* Cost is bounded per case by `MAX_SUPERVISOR_STEPS` × (1 routing call + `MAX_TOOL_ITERATIONS` tool
  rounds + 1 summarisation) + 1 report call. That ceiling is configuration, so it can be tuned per
  environment and enforced in review.
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
  measuring real routing quality needs an offline eval set of cases with expected routes, which is
  the first thing I would build next.
* No authentication, rate limiting or PII redaction — deliberately out of scope for an exercise, and
  listed in the README's hardening table.
