# Open PR Triage — agno-agi/agno

**Date:** 2026-02-16
**Total open PRs:** 293

---

## Duplicates to Close

These PRs have a better counterpart open. Close with a comment pointing to the kept PR.

### 1. `get_member_id` UUID priority

Bug: When a member has a UUID `id` and a `name`, the name wins. Should be the other way around.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| **KEEP** | [#6336](https://github.com/agno-agi/agno/pull/6336) | fix: prioritize explicit id over name in get_member_id | kausmeows | Clean rewrite, updated tests, `Optional[str]` return type |
| CLOSE | [#6434](https://github.com/agno-agi/agno/pull/6434) | fix: prioritize UUID id over name in get_member_id | themavik | Same bug, minimal reorder, no test updates |

**Close comment for #6434:**
> Closing as duplicate of #6336 which addresses the same bug with a more complete fix including updated tests.

---

### 2. Datetime serialization in Postgres session storage

Bug: `datetime` objects in session data fail JSON serialization when writing to PostgreSQL JSONB columns.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| **KEEP** | [#6363](https://github.com/agno-agi/agno/pull/6363) | fix: datetime JSON serialization for PostgreSQL session storage | kausmeows | Fixes at SQLAlchemy engine level (catches all paths), 292-line test file |
| CLOSE | [#6436](https://github.com/agno-agi/agno/pull/6436) | fix: handle datetime serialization in session data for PostgreSQL storage | themavik | Fixes at `to_dict()` level only, misses other write paths |
| CLOSE | [#6437](https://github.com/agno-agi/agno/pull/6437) | fix: handle SessionSummary and datetime serialization in session storage | themavik | Near-clone of #6436 with minor additions, same architectural issue |

**Close comment for #6436 and #6437:**
> Closing as duplicate of #6363 which fixes this at the SQLAlchemy engine level, covering all serialization paths (not just `to_dict()`), and includes comprehensive tests.

---

### 3. OpenAI reasoning content extraction

Bug: Gemini via OpenRouter returns reasoning in a `reasoning` attribute instead of `reasoning_content`.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| CLOSE | [#5332](https://github.com/agno-agi/agno/pull/5332) | [fix] Support reasoning attribute in OpenAI-compatible API responses | krishnarathore12 | **Already fixed on `main`** — code already handles both `reasoning_content` and `reasoning` |
| CLOSE | [#4177](https://github.com/agno-agi/agno/pull/4177) | [fix] Fixed openai api reasoning from reasoning_content to reasoning | MarcoFerreiraPerson | **Already fixed on `main`** + low quality (includes committed junk files) |

**Close comment:**
> This bug has already been fixed on `main`. The code at `models/openai/chat.py` now handles both `reasoning_content` and `reasoning` attributes. Closing.

---

### 4. OpenRouter reasoning extraction

Bug: Reasoning data from Gemini/other models via OpenRouter not extracted properly.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| **KEEP one** | [#5802](https://github.com/agno-agi/agno/pull/5802) | feat: support Gemini 3 Thought Signatures in OpenRouter adapter | iplus26 | More comprehensive — adds `ReasoningConfig`, preserves reasoning for tool calling, integration tests |
| **OR KEEP** | [#6111](https://github.com/agno-agi/agno/pull/6111) | fix: openrouter gemini reasoning | uzaxirr | More focused extraction fix, from a regular contributor |

**Note:** These overlap on `models/openrouter/openrouter.py`. Need to check which one is less stale and applies cleanly. #6111 author (uzaxirr) is a more active contributor. Pick one, close the other.

---

### 5. Agent metadata merge

Bug: Metadata passed to `agent.run()` not correctly set on `run_context`.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| CLOSE | [#6340](https://github.com/agno-agi/agno/pull/6340) | fix: ensure metadata passed to agent.run() is set on run_context | kausmeows | **Already fixed on `main`** — metadata handling refactored into `_run_options.py` |
| CHECK | [#6452](https://github.com/agno-agi/agno/pull/6452) | fix: runtime metadata should take precedence over agent defaults | themavik | May address a *different* aspect (merge direction). Verify if the `_run_options.py` refactor also fixed precedence |

**Close comment for #6340:**
> This bug has been fixed on `main` via the refactor into `_run_options.py`. The `resolve_run_options()` function now correctly handles metadata merging and assignment to `run_context`. Closing.

---

### 6. MCP header_provider validation

Bug: Passing `header_provider` with incompatible transport (e.g., `stdio`) silently ignored.

| Action | PR | Title | Author | Reason |
|--------|----|-------|--------|--------|
| CLOSE | [#5988](https://github.com/agno-agi/agno/pull/5988) | refactor: Fail explicitly when header_provider is incompatible with transport | bhatt-neel-dev | **Already handled on `main`** (warning + ignore approach). The PR wanted ValueError instead, which is a behavior change that should be a deliberate decision |

**Close comment:**
> The incompatible transport case is already handled on `main` (logs a warning and ignores the header_provider). The proposed ValueError approach would be a breaking change. Closing.

---

## PRs Already Fixed on `main` — Close

These address bugs that no longer exist in the current codebase:

| PR | Title | Why it's already fixed |
|----|-------|----------------------|
| [#6340](https://github.com/agno-agi/agno/pull/6340) | fix: ensure metadata passed to agent.run() is set on run_context | Refactored into `_run_options.py` |
| [#5332](https://github.com/agno-agi/agno/pull/5332) | [fix] Support reasoning attribute in OpenAI-compatible API responses | `chat.py` already handles `reasoning` attr |
| [#4177](https://github.com/agno-agi/agno/pull/4177) | [fix] Fixed openai api reasoning | Same as #5332, also low quality |
| [#5988](https://github.com/agno-agi/agno/pull/5988) | refactor: Fail explicitly when header_provider is incompatible | Already handled via warning + ignore |

---

## Low-Hanging Bug Fixes — Merge Candidates

These are still relevant on `main`, verified by reading current code.

### Tier 1: Smallest, safest fixes

| PR | Title | Size | What's broken |
|----|-------|------|---------------|
| [#6175](https://github.com/agno-agi/agno/pull/6175) | Cast session_name to TEXT in PostgresDb.rename_session() | +2/-2 | `to_jsonb()` fails on untyped bind param |
| [#6336](https://github.com/agno-agi/agno/pull/6336) | Prioritize explicit id over name in get_member_id | +31/-17 | UUID ids deprioritized below names, breaks DB sync |
| [#6193](https://github.com/agno-agi/agno/pull/6193) | Pin python-multipart>=0.0.22 | 2 lines | Security vulnerability (Dependabot alert) |
| [#6363](https://github.com/agno-agi/agno/pull/6363) | Datetime JSON serialization for Postgres sessions | +29 +tests | datetime objects crash session storage |
| [#6250](https://github.com/agno-agi/agno/pull/6250) | Fix cache key serialization for structured outputs | +9/-1 | Pydantic model class not JSON-serializable |
| [#6300](https://github.com/agno-agi/agno/pull/6300) | Enable native structured outputs for LMStudio | +1/-1 | 1-line flag flip |
| [#6438](https://github.com/agno-agi/agno/pull/6438) | Prevent reasoning_content duplication | +3/-5 | Append should be reassign |
| [#6392](https://github.com/agno-agi/agno/pull/6392) | Handle reasoning_text.delta events in streaming | +3/-3 | Wrong event type checked |

### Tier 2: Slightly larger but correct

| PR | Title | Size | What's broken |
|----|-------|------|---------------|
| [#6495](https://github.com/agno-agi/agno/pull/6495) | HITL pause session state loss | +132/-8 | `run_context` not passed to pause handlers, session_state lost |
| [#5881](https://github.com/agno-agi/agno/pull/5881) | Pass response_format to LiteLLM completion | +378/-6 | `response_format` silently dropped, structured outputs broken |
| [#5319](https://github.com/agno-agi/agno/pull/5319) | Add missing id to OpenAI Responses tool output | +10/-1 | API requires `id` field, currently missing |
| [#6104](https://github.com/agno-agi/agno/pull/6104) | MCP header_provider not passed during connect | +4 | Headers never sent to streamablehttp_client() |
| [#6312](https://github.com/agno-agi/agno/pull/6312) | stop_after_tool_call in ToolExecution.is_paused | +19/-0 | stop_after_tool_call + output_schema causes JSON parse error |
| [#5755](https://github.com/agno-agi/agno/pull/5755) | Remove include_answer from Tavily get_search_context() | 1 line | Tavily SDK throws on unsupported kwarg. Branch has drifted, needs rebase |
