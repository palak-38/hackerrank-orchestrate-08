# Domain glossary

The words this codebase uses for its own concepts. Code, docs, and reviews use these terms as
defined here; when a new concept gets a name, add it.

**Message** — one incoming WhatsApp message to route, from `messages.csv` (`IncomingMessage`). It
belongs to one receiving user; everything about it is judged from that user's point of view.

**Action** — what happens to a message: `notify` (interrupt now), `digest` (show later), or `mute`
(suppress).

**Routing decision** — the full answer for one message: action, message_type, reason, confidence,
and evidence IDs, plus provenance (decider, provider, model, prompt version). One row of
`output.csv` (`RoutingDecision`). Made only in `routing.py`.

**Routing context** — everything known about a message when it is decided: the user, group,
business, media analysis, retrieved evidence, features, and the scam check's result
(`MessageRoutingContext`).

**Evidence** — earlier messages from the *same user's* history that support a decision, cited by ID
in `evidence_message_ids`, or `none`.
- *Retrieved evidence*: what SQL + vector retrieval found before any decider ran.
- *Evidence pick*: IDs a decider chose itself. Used only if every ID was shown to that decider;
  otherwise the retrieved evidence stands (an *evidence fallback*, recorded in the trace).

**Scam check** — the deterministic rules in `safety_gate.py`. Puts every message in one tier:
- *certain*: muted as `scam` with no LLM call. Prompt injection is always here.
- *suspicious*: decided by the decider, whose prompt carries the fired rules and asks it to
  confirm or reject.
- *clear*: no rule fired.

**Decider** — the swappable step that turns a routing context into a decision (`deciders/`).
Everything around it — scam check, evidence rule, provenance, cache — is the same for every decider.
- `single`: one structured LLM call, with provider failover.
- `agent`: an Agents SDK tool loop that may look up more history and picks its own evidence.

**Trace** — per-message record of how a decision was reached: gate tier, model requests, tokens,
tool calls, evidence shown, any fallback. Never stores message text (`trace.py`).

**Prompt version** — `PROMPT_VERSION` in `config.py`. Cached decisions are reused only under the
same prompt version *and* the same decider.
