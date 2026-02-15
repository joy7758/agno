# Agno Bug Triage — Oldest to Latest

> Systematic deep review of all 129 open bugs, analyzed from oldest to latest.
> Each bug is traced through the actual codebase on `main`, with conflicting perspectives and a final verdict.

---

## Bug #3964 — Gemini `response_mime_type` + Tools = 400 Error

| Field | Value |
|-------|-------|
| **Created** | 2025-07-24 |
| **Labels** | `bug`, `stale` |
| **Author** | @JoeMukherjeeAssessli |
| **Env** | Windows, agno 1.7.5, Gemini 2.5/2.0 flash |
| **Community PR** | #4082 (OPEN, never reviewed) |
| **Duplicates** | Related to #4298 (Gemini cache + tools) |

### Problem

Using Gemini models with both structured output (`output_schema` or `response_mime_type: "application/json"`) AND tools/function calling causes:
```
400 INVALID_ARGUMENT: Function calling with a response mime type: 'application/json' is unsupported
```

### Code Trace

**Path 1 (raw `generation_config`):**
- `gemini.py:241-245` — user's config becomes `config` dict (includes `response_mime_type`)
- `gemini.py:331-332` — tools added unconditionally
- Both present → 400

**Path 2 (`output_schema` — the "proper" way):**
- `_response.py:844-847` — Gemini has `supports_native_structured_outputs = True` (line 79), so raw Pydantic class returned as `response_format`
- `gemini.py:274` — sets `config["response_mime_type"] = "application/json"` unconditionally
- `gemini.py:331-332` — tools added unconditionally
- Both present → same 400

**No guard exists anywhere** to strip `response_mime_type` when tools are detected.

### Perspectives

**"It's real"**: Line 274 is unconditional, no guard, no tests, 20 commits since filing never fixed it, both API paths fail, community confirmed independently.

**"Maybe not"**: Google may have updated the API for newer models; only fails when combining output_schema + tools specifically.

**Rebuttal**: Even if API evolved, no graceful fallback exists. And output_schema + tools is an extremely common pattern.

### Verdict: CONFIRMED BUG — Still Active on `main`

**Fix**: `gemini.py:get_request_params()`, after line 332:
```python
if config.get("tools") and config.get("response_mime_type"):
    config.pop("response_mime_type")
```

**Action**: Fix the bug, add unit test, review/close community PR #4082.

---

## Bug #3968 — Bedrock Model Not Refreshing AWS Token

| Field | Value |
|-------|-------|
| **Created** | 2025-07-24 |
| **Labels** | `bug` |
| **Author** | @sirianni |
| **Env** | EKS production, short-lived IAM credentials |
| **Agno Response** | "fix out asap" (Oct 2025) — never shipped |

### Problem

The `AwsBedrockClaude` model extracts AWS credentials (access key, secret key, session token) as raw strings from a boto3 `Session` ONCE at client creation, then caches the client forever. When short-lived credentials expire (~1 hour for IAM roles, EKS pods), all subsequent API calls fail with:
```
403: The security token included in the request is expired
```

### Code Trace

**Credential snapshot** — `claude.py:54-62`:
```python
credentials = self.session.get_credentials()
client_params = {
    "aws_access_key": credentials.access_key,      # SNAPSHOT
    "aws_secret_key": credentials.secret_key,      # SNAPSHOT
    "aws_session_token": credentials.token,         # SNAPSHOT (expires!)
}
```

**Sync client cached** — `claude.py:96-122`: Has `is_closed()` check but that doesn't detect expired tokens. Client cached forever with stale credentials.

**Async client cached** — `claude.py:124-152`: **No `is_closed()` check at all** (line 131 just checks `is not None`). Even worse than sync.

**Comparison**: Regular Anthropic Claude (`anthropic/claude.py:386-391`) at least has `is_closed()` on async. Bedrock doesn't.

**Note**: The newer `bedrock.py` (non-Anthropic) async path (`line 167`) creates a new client via `self.async_session.client()` each call — slightly better but its sync path has the same cache problem.

### Perspectives

**"It's real"**: Credential strings extracted once (line 56-62), both clients cached forever, no TTL/refresh logic anywhere, 15 commits since filing none address it, Agno team acknowledged 4 months ago with no fix.

**"Maybe not"**: Only affects short-lived credentials (IAM roles, EKS); static env var credentials don't expire. Users can manually set `model.client = None` to force recreation.

**Rebuttal**: IAM best practice (and AWS security guidelines) mandate temporary credentials. Manual client invalidation is not a real workaround for production. Every major AWS deployment uses roles.

### Verdict: CONFIRMED BUG — Still Active on `main`

**Fix options**:
1. Re-call `session.get_credentials()` on every `_get_client_params()` invocation (simple, slightly more overhead)
2. Add TTL-based refresh: check `credentials._expiry_time` and recreate client when near expiry
3. Add `is_closed()` check to async client (line 131) at minimum

**Action**: Implement credential refresh for boto3 session-based clients. Add `is_closed()` to async path.

---

## Bug #3980 — Token Inflation ~150x with OpenRouter

| Field | Value |
|-------|-------|
| **Created** | 2025-07-25 |
| **Labels** | `bug` |
| **Author** | @SametHaymana |
| **Env** | OpenRouter + Claude (via OpenAIChat), PostgreSQL memory |
| **Confirmed by** | @JasonLovesDoggo (514 instances), @udemirezen |
| **Related** | #5741 (exponential session.runs growth), #5838 (session data growth) |

### Problem

User reports requesting ~34K tokens worth of content but the API error says 5,028,244 tokens were sent. That's a **~150x inflation**. Multiple users independently confirmed this.

### Code Trace

**Root cause: Quadratic history storage growth**

Prior to the fix, `store_history_messages` defaulted to `True`. This caused:
- Run 1 stores: its own N messages
- Run 2 stores: its own N messages + Run 1's N messages = 2N
- Run 3 stores: N + 2N = 3N (own + all previous)
- After K runs: storage = N + 2N + 3N + ... + KN = **O(K²)**

When `add_history_to_context=True` (or `enable_user_memories=True` which enables it), ALL these stored messages are retrieved and sent to the model. A conversation with 20 runs could easily reach millions of tokens.

**History retrieval** — `agent.py:519-520`:
```python
if self.num_history_messages is None and self.num_history_runs is None:
    self.num_history_runs = 3  # default: last 3 runs
```

With `store_history_messages=True` (old default), even 3 runs of history could be enormous because each run contained the full accumulated history of all prior runs.

**The fix** — `agent.py:213` (now on `main` via v2.5 refactor):
```python
store_history_messages: bool = False  # Changed from True
```

### Perspectives

**Perspective A: "Fixed on main"**

1. `store_history_messages` now defaults to `False` on `main` (line 213)
2. This was delivered via v2.5 Phase 1 (`#6429`) and the specific fix PR `#6315`
3. With `False`, each run stores only its own messages — O(N) not O(N²)
4. The default `num_history_runs=3` with non-duplicated runs keeps context bounded
5. The bug was filed on agno 1.7.5 (July 2025) — the fix is in v2.5+ (Feb 2026)

**Perspective B: "Residual risk remains"**

1. Users who explicitly set `store_history_messages=True` still hit this
2. Memory injection (`enable_user_memories=True`) adds memory content to EACH system message — if history retrieves multiple runs, each run's system message includes the same memories, causing duplication
3. `store_tool_messages=True` (still the default) means tool results (which can be very large) are stored and re-sent in history
4. `num_history_runs` defaults to 3 — with large tool results this can still inflate tokens significantly
5. No compression by default — `compress_tool_results` and `CompressionManager` exist but are opt-in

**Assessment of Perspective B:**

Points 2-5 describe **linear** growth (not quadratic), which is expected behavior — you asked for history, you get history. The 150x inflation reported in the original bug is firmly in O(N²) territory and IS fixed. The residual concerns are optimizations, not bugs.

### Verdict: LIKELY FIXED on `main` (v2.5+)

The quadratic storage growth that caused 150x token inflation has been fixed by changing `store_history_messages` default to `False`. Users on versions < 2.5 are still affected.

**Remaining risks** (not bugs, just optimization opportunities):
- Large tool results in history (`store_tool_messages=True` default)
- Memory duplication in system messages across historical runs

**Action**: Can likely be closed with a note pointing to v2.5 upgrade. Optionally link to #5741 and #5838 as related/duplicate.

---

## Bug #4184 — Cross-Provider Session Incompatibility (messages[N].content expected object got string)

| Field | Value |
|-------|-------|
| **Created** | 2025-08-12 |
| **Labels** | `bug`, `stale` |
| **Author** | @MarcosPTProenca |
| **Env** | Docker Linux, Python 3.10, agno 1.7.2, OpenAI + Gemini |
| **Confirmed by** | @aislanmaia, @arhen, @agoswami84 |
| **Duplicates** | #5713 (same root cause, filed Dec 2025) |

### Problem

When using PostgreSQL storage with session history, and **switching between model providers** (Gemini → OpenAI) in the same session, OpenAI rejects the replayed history:
```
Invalid type for 'messages[7].content[0]': expected an object, but got a string instead.
```

Also confirmed: happens when using **different providers** for `model` and `output_model`, or when replaying ANY Gemini-originated session through OpenAI.

### Code Trace

**The format mismatch:**

1. **Message model** — `message.py:64`: `content: Optional[Union[List[Any], str]]`
   - Content can be a list (Gemini format) or string (OpenAI format)
   - Stored as-is in PostgreSQL session storage

2. **Gemini stores tool results** in its native format (list of Part objects, serialized as dicts)

3. **OpenAI message formatting** — `openai/chat.py:326-330`:
   ```python
   tool_result = message.get_content(...)
   message_dict = {"role": ..., "content": tool_result, ...}
   ```
   This passes whatever format was stored — if it was Gemini's list format, OpenAI chokes on it.

4. **No normalization layer exists** — there is no cross-provider message format normalization. Each model's `_format_messages()` trusts that `message.content` is in its expected format. When history crosses providers, this trust is violated.

**Related**: Issue #5713 describes the exact same root cause with Firestore instead of PostgreSQL.

### Perspectives

**Perspective A: "This is a real architectural gap"**

1. Agno supports provider-agnostic sessions — same `session_id` across different models is a core feature
2. No message normalization layer exists between providers
3. 4+ independent users hit this, including one with 514 instances (@JasonLovesDoggo, though that may overlap with #3980)
4. The `Message` model stores provider-specific formats without normalization
5. @arhen even built a custom workaround (extracting content before sending to OpenAI/Claude)
6. Issue #5713 (Dec 2025) describes the same bug — this has persisted 6+ months

**Perspective B: "It's a known limitation, not a bug"**

1. Cross-provider sessions are an edge case — most users stick to one provider
2. Different providers have fundamentally different message formats (function calling, tool results, multimodal content)
3. Full normalization is a major architectural effort — would need a canonical message format and per-provider serializers/deserializers
4. Users can work around it by using separate sessions per provider
5. The original reporter was on agno 1.7.2 — the v2.x rewrite may have improved message handling

**Assessment:**

This IS a real bug — Agno's value proposition is provider-agnostic agents. If sessions break when switching providers, that's a fundamental issue. However, the fix is architecturally significant (requires a message normalization layer). The fact that #5713 was filed 4 months later with the same root cause confirms it's still active.

### Verdict: CONFIRMED BUG — Still Active on `main`

No cross-provider message normalization exists in the codebase. Switching providers within a session breaks tool message replay.

**Fix**: Add a normalization step in session message retrieval that converts stored messages to the target provider's format. At minimum, tool result `content` should be normalized to string format before sending to OpenAI.

**Action**: Keep open. Link #5713 as duplicate. This requires an architectural fix.

---

## Bug #4298 — Gemini Cache + Tools/System Instruction = 400 Error

| Field | Value |
|-------|-------|
| **Created** | 2025-08-21 |
| **Labels** | `bug` |
| **Author** | @erictom97 |
| **Env** | Gemini API, agno with cached_content |
| **Related** | #3964 (same file, different constraint) |

### Problem

When using Gemini's cached content feature (`cached_content`) alongside an Agent that has tools, the Gemini API rejects the request:
```
400 INVALID_ARGUMENT: Tool config, tools and system instruction should not be set
in the request when using cached content.
```

The Gemini API requires that when `cached_content` is provided, the request must NOT include `system_instruction`, `tools`, or `tool_config` — because those are already locked into the cache at creation time.

### Code Trace

**`cached_content` added unconditionally** — `gemini.py:266`:
```python
config.update({
    ...
    "cached_content": self.cached_content,
})
```

**`system_instruction` added unconditionally** — `gemini.py:270-271`:
```python
if system_message is not None:
    config["system_instruction"] = system_message
```

**`tools` added unconditionally** — `gemini.py:327-332`:
```python
if builtin_tools:
    config["tools"] = builtin_tools
elif tools:
    config["tools"] = [format_function_definitions(tools)]
```

**`tool_config` added unconditionally** — `gemini.py:334-344`:
```python
if tool_choice is not None:
    config["tool_config"] = ...
```

**No guard exists** — the `None`-stripping at line 346 only removes keys with `None` values. Since `cached_content`, `system_instruction`, `tools`, and `tool_config` are all set to non-None values, all four end up in the final config.

### Perspectives

**"It's real"**: Line 266 sets `cached_content`, and lines 271, 327-332, 334-344 add conflicting keys with zero awareness that a cache is present. No guard, no test, no documentation warning. The cookbook example (`file_upload_with_cache.py`) works only because it creates the Agent WITHOUT tools — the moment you add tools, it breaks. This is a limitation that prevents a very common use case: caching a large system prompt + using tools.

**"Maybe not"**: The user could structure their cache to include the tools/system instruction at cache creation time, and then simply not add tools to the Agent. The error message from Google is clear, and the user can work around it by design. Also, this is arguably a Gemini API limitation, not an Agno bug.

**Rebuttal**: Agno's Agent class automatically adds tools and system instructions — the user doesn't control what gets sent to the API at the config level. When a user creates `Agent(model=Gemini(cached_content=cache), tools=[SomeTool()])`, Agno should handle the conflict gracefully. The cache already has the tools/system baked in, so Agno should strip them from the request. Not doing so is an Agno bug.

### Verdict: CONFIRMED BUG — Still Active on `main`

**Evidence:**
- `gemini.py:266` adds `cached_content`
- Lines 271, 327-332, 334-344 unconditionally add `system_instruction`, `tools`, `tool_config`
- No guard strips conflicting keys when `cached_content` is present
- Reproduction test FAILS: config contains all four keys simultaneously

**Fix**: `gemini.py:get_request_params()`, before line 346 (the None-strip):
```python
if config.get("cached_content"):
    config.pop("system_instruction", None)
    config.pop("tools", None)
    config.pop("tool_config", None)
```

**Action**: Fix the bug alongside #3964 (same method, same file). Add unit test.

---

## Bug #4430 — Bedrock Structured Output: tool_choice Ignored

| Field | Value |
|-------|-------|
| **Created** | 2025-09-04 |
| **Labels** | `bug` |
| **Author** | @ArlindNocaj |
| **Env** | Bedrock (Nova + Claude models) |
| **Confirmed by** | @hashbrown, @sushantgundla |
| **Agno Response** | "We'll take a look very soon!" (Sep 2025) — never shipped |

### Problem

When using structured output (`output_schema`) with Bedrock models, the Bedrock Converse API receives `toolChoice: auto` (implicit) instead of `toolChoice: {"tool": {"name": "MyOutputModel"}}`. This means the model is free to ignore the output schema tool and respond with plain text.

### Code Trace

**`invoke()` accepts but ignores `tool_choice`** — `bedrock.py:440-458`:
```python
def invoke(self, ..., tool_choice=None, ...):
    tool_config = None
    if tools:
        tool_config = {"tools": self._format_tools_for_request(tools)}
    #                  ^^^^^^^^ only "tools", no "toolChoice"
```

The `tool_choice` parameter is in the method signature (line 446) but is **never referenced** in the method body. The `toolConfig` dict only gets `"tools"` — the `"toolChoice"` key is never added.

**Same issue in `invoke_stream()`** — `bedrock.py:493-507`: identical code, same omission.

**Same issue in `ainvoke()`** — `bedrock.py:546` and `ainvoke_stream()` — `bedrock.py:598`.

**Comparison with Gemini**: `gemini.py:334-344` does handle `tool_choice` and maps it to `tool_config`. Bedrock doesn't.

### Perspectives

**"It's real"**: The `tool_choice` parameter is accepted in the method signature but completely ignored in all 4 invoke variants (sync, stream, async, async stream). Multiple users confirmed this in production. The Agno team acknowledged 5 months ago but never fixed it.

**"Maybe not"**: Bedrock models generally follow instructions well enough that `auto` tool choice still triggers the structured output tool most of the time. The bug is really about reliability, not a hard failure.

**Rebuttal**: "Most of the time" is not acceptable for production structured output. The whole point of `toolChoice: {"tool": {"name": ...}}` is to **guarantee** the model uses the tool. Without it, structured output is non-deterministic.

### Verdict: CONFIRMED BUG — Still Active on `main`

**Evidence:**
- `tool_choice` parameter accepted but never used in `invoke()` (line 446 vs 456-458)
- Same omission in all 4 invoke variants
- Reproduction test FAILS: `toolConfig` only has `"tools"`, no `"toolChoice"`

**Fix**: `bedrock.py:invoke()`, after line 458:
```python
if tools:
    tool_config = {"tools": self._format_tools_for_request(tools)}
    if tool_choice is not None:
        tool_config["toolChoice"] = tool_choice
```

Apply same fix to `invoke_stream()`, `ainvoke()`, and `ainvoke_stream()`.

**Action**: Fix the bug in all 4 invoke methods. Add unit test.

---

## Bug #4540 — Performance Regression in Knowledge.add_content_async

| Field | Value |
|-------|-------|
| **Created** | 2025-09-11 |
| **Labels** | `bug` |
| **Author** | @zhrli |
| **Env** | Linux, agno 2.0.2, LanceDB |
| **Confirmed by** | @chyhkae, @dongtl4, @jesalg |

### Problem

After upgrading from agno 1.8.x to 2.0.x, knowledge ingestion became extremely slow. Each chunk was embedded individually (separate HTTP request per chunk), and the default chunk size was accidentally set to 100 characters instead of 5000.

### Code Trace

**Root cause 1**: `FixedSizeChunking.__init__()` had `chunk_size=100` in v2.0.0 release (a placeholder that was never updated). A 10K doc → 100 chunks instead of 2.

**Root cause 2**: No batch embedding — each chunk triggered a separate API call to the embedder.

### Fix Verification

1. **Chunk size fixed**: `chunk_size=5000` — commit `3733e1721` (Sep 10, 2025)
2. **Batch embedding added**: `enable_batch` flag on embedder + `async_get_embeddings_batch_and_usage()` — commit `6556ef9fa` (Oct 1, 2025)
3. **VectorDB batch upsert**: Documents processed in 100-doc batches

### Verdict: PARTIALLY FIXED on `main`

**Root cause 1 (chunk_size=100): FIXED.** Default now 5000.

**Root cause 2 (individual embedding calls): PARTIALLY FIXED.**
- Batch embedding exists but `enable_batch=False` by default
- Async path uses `asyncio.gather()` for concurrent embedding (better but still N API calls)
- Sync path still embeds one-at-a-time in a loop (`lancedb/lance_db.py:315-326`)
- Users must explicitly set `embedder.enable_batch=True` to get true batch embedding

The reporter's core complaint — "hundreds of API calls for a large document" — is significantly reduced (2 chunks vs 100) but not fully solved for users on the sync path who don't know about `enable_batch`.

**Action**: Partially close. The chunk_size regression is fixed. The batch embedding default is a design choice, but should be documented better. Consider changing `enable_batch` default to `True`.

---

## Bug #4573 — AgentOS Not Seeing FastMCP Tools

| Field | Value |
|-------|-------|
| **Created** | 2025-09-12 |
| **Labels** | `bug` |
| **Author** | @EloisaMazzocco00 |
| **Env** | macOS, agno 2.0.3, FastMCP 2.12.2 |
| **Confirmed by** | @JasonLovesDoggo |

### Problem

When connecting a FastMCP server to an Agent in AgentOS, no tools were detected. The MCP server worked fine with `fastmcp.Client` directly, but tools list was empty in AgentOS.

### Code Trace

**Root cause**: MCP connection lifecycle issue. `MCPTools.build_tools()` is async and requires an active `ClientSession` (established via `async with MCPTools(...)`). In the original code, the Agent/AgentOS didn't properly manage this async lifecycle — MCPTools weren't self-connecting within the agent's run cycle, so tools were never populated.

**Fix**: Commit `ba81b82da` (Oct 28, 2025) — "MCPTools now self-connect inside agents like they do in AgentOS." This was a massive MCP overhaul (1060 insertions, 802 deletions) that:
- Restructured MCP into `mcp/mcp.py`, `mcp/multi_mcp.py`, `mcp/params.py`
- Added `refresh_mcp_tools` flag on Agent
- Made MCPTools self-connect during agent runs
- Added connection error resilience

**44 additional MCP commits** since the bug was filed, including:
- `268a1f37e` — prevent MCP connection failures from causing 500 errors
- `ed8dfb5cf` — fix double MCP lifespan in AgentOS
- `ab0b21f16` — default to streamable-http transport when url is present

### Perspectives

**"It's fixed"**: The core issue (MCP lifecycle not integrated with agent lifecycle) was addressed in Oct 2025. 44 subsequent MCP commits have further hardened the connection handling. The `build_tools()` path works correctly (verified by test).

**"Maybe not fully"**: The original reporter never confirmed the fix worked for their use case. AgentOS-specific edge cases may still exist — the most recent fix (`ed8dfb5cf`, Feb 2026) was still fixing MCP lifespan issues in AgentOS just days ago.

### Verdict: LIKELY FIXED on `main`

The core framework issue is resolved. `build_tools()` correctly populates functions from an MCP session. The AgentOS-specific lifecycle has been fixed multiple times. However, no confirmation from the original reporter.

**Action**: Close with a note pointing to the MCP overhaul. If users still experience this on latest, it's a new bug.

---

## Bug #4688 — Generator exiting early / connection leaks in delegate_task_to_member

**Issue**: https://github.com/agno-agi/agno/issues/4688
**Filed**: ~Sep 2025
**Reporter**: AgentOS team (internal)

### Problem

`adelegate_task_to_member` used `break` inside an `async for` loop over the member's response stream. Breaking out of an `async for` triggers `GeneratorExit` on the underlying async generator, which:
1. Causes the generator to abort mid-iteration
2. Prevents proper cleanup of the streaming connection
3. Led to connection leaks and session save failures in AgentOS

### Code Trace

**`libs/agno/agno/team/_default_tools.py:665`**:
```python
# Do NOT break out of the loop, AsyncIterator need to exit properly
```

The fix replaced `break` with `continue`, letting the async iterator exhaust naturally. The generator completes its full lifecycle, connections close properly, and session state is saved.

### Test

**`tests/unit/team/test_delegate_no_early_break.py`** — AST-based regression guard that parses the source code of `adelegate_task_to_member` and asserts no `break` statement exists inside any `async for` loop. **PASSES** (fix is in place).

### Perspectives

**"It's fixed"**: The `break` was replaced with `continue` at line 665. The comment explicitly documents why. The AST test confirms no `break` exists in the async for loop.

**"Could regress"**: A future developer might not understand why `continue` is used instead of `break` (the natural instinct). The comment helps, but without the AST test guard, someone could reintroduce the `break` thinking it's a performance optimization.

### Verdict: FIXED on `main`

The generator lifecycle issue is fully resolved. The AST-based regression test ensures it stays fixed.

**Action**: Close.

---

## Batch-Verified Closures (Cross-Referenced via Commit History)

> The following 5 bugs were identified by cross-referencing 129 open issues against 894 recent commits,
> then verifying each fix by reviewing the full PR diff, reading all issue comments, and writing
> AST-based regression tests. All tests pass on `main` (rebased 2026-02-15).

---

## Bug #5754 — Tavily `include_answer` TypeError in `web_search_with_tavily`

| Field | Value |
|-------|-------|
| **Created** | 2025-12-11 |
| **Labels** | `bug` |
| **Author** | @mklrlnd424 |
| **Fix PR** | #5755 (commit `03da24873`) |

### Problem

`web_search_with_tavily()` passed `include_answer=self.include_answer` to `client.get_search_context()`, which internally also passes `include_answer` to `_search()`, causing `TypeError: got multiple values for keyword argument 'include_answer'`.

### Fix

Removed `include_answer` kwarg from the `get_search_context()` call. The `search()` method (used by `web_search_using_tavily`) correctly still passes it.

### Regression Test

`tests/unit/tools/test_tavily_include_answer.py` — AST-based check that `get_search_context()` call has no `include_answer` kwarg. **PASSES.**

### Verdict: FIXED — Closed

---

## Bug #5173 — Milvus Missing `radius`/`range_filter` Support

| Field | Value |
|-------|-------|
| **Created** | 2025-10-24 |
| **Labels** | `bug` |
| **Author** | @starzeng |
| **Fix PR** | #5732 |

### Problem

Milvus `search()` and `async_search()` didn't accept `search_params`, so users couldn't pass `radius` and `range_filter` to fine-tune search behavior at runtime.

### Fix

Added `search_params` kwarg to both `Milvus.search()` and `Milvus.async_search()`.

### Regression Test

`tests/unit/vectordb/test_milvus_search_params.py` — AST-based check for `search_params` parameter. **PASSES.**

### Verdict: FIXED — Closed

---

## Bug #5858 — Crawl4AI No Proxy Setting

| Field | Value |
|-------|-------|
| **Created** | 2025-12-25 |
| **Labels** | `bug` |
| **Author** | @lwtwan |
| **Fix PR** | #5859 |

### Problem

`Crawl4aiTools` had no way to configure a proxy. Users had to manually modify SDK source code.

### Fix

Added `proxy_config` parameter to `Crawl4aiTools.__init__` and stored it as `self.proxy_config`.

### Regression Test

`tests/unit/tools/test_crawl4ai_proxy.py` — AST-based check for `proxy_config` parameter and `self.proxy_config` assignment. **PASSES.**

### Verdict: FIXED — Closed

---

## Bug #5329 — Missing `reasoning_content` with OpenRouter Gemini

| Field | Value |
|-------|-------|
| **Created** | 2025-11-13 |
| **Labels** | `bug` |
| **Author** | @fehmisener |

### Problem

When using Gemini models via OpenRouter, `reasoning_content` was `None` despite `reasoning_tokens` being non-zero. OpenRouter returns reasoning in a `reasoning` attribute instead of `reasoning_content`, and the parser only checked for the latter.

### Fix

OpenAI chat parser (`_parse_provider_response` and `_parse_provider_response_delta`) now checks for `.reasoning` attribute as fallback alongside `.reasoning_content`.

### Regression Test

`tests/unit/models/openai/test_openrouter_reasoning_content.py` — AST-based check that both parse methods handle `.reasoning` fallback. **PASSES.**

### Verdict: FIXED — Closed

---

## Bug #5334 — AgentOS MCP + AWS Lambda Incompatibility

| Field | Value |
|-------|-------|
| **Created** | 2025-11-06 |
| **Labels** | `bug` |
| **Author** | @jakubno |
| **Env** | AWS Lambda, Python 3.13, agno >= 2.2.8 |

### Problem

`StreamableHTTPSessionManager.run()` was called multiple times in stateless Lambda environments, raising `RuntimeError: can only be called once per instance`.

### Fix

MCP lifespan management refactored to avoid double initialization. The `_make_app` method no longer references `mcp_lifespan` directly.

### Regression Test

`tests/unit/os/test_agentos_mcp_lifespan.py` — AST-based check that `_make_app` doesn't reference `mcp_lifespan`. **PASSES.**

### Verdict: FIXED — Closed

---

## NEEDS_TEST Batch Results (Cross-Referenced via Commit History)

> The following 4 bugs had matching PRs but required deeper verification.
> Assessed by tracing current code on `main`, checking filing dates vs PR merge dates,
> and analyzing whether the fix actually addresses the reported issue.

---

## Bug #6327 — Datetime Not JSON-Serializable in PostgreSQL Sessions

| Field | Value |
|-------|-------|
| **Created** | 2026-01-30 |
| **Labels** | `bug` |
| **Fix PR** | #5694 (commit `33c31883`) |

### Problem

Saving agent sessions to PostgreSQL failed with `TypeError: Object of type datetime is not JSON serializable` because `RunRequirement.created_at` (a `datetime` object) was passed directly to `json.dumps()`.

### Fix

`RunRequirement.to_dict()` (line 136) now converts `created_at` to ISO string via `.isoformat()`. `from_dict()` (lines 169-178) parses ISO strings back to datetime. Round-trip through `json.dumps()`/`json.loads()` works correctly.

### Regression Test

`tests/unit/run/test_run_requirement_datetime_serialization.py` — Tests `to_dict()` produces JSON-serializable output and `from_dict()` round-trips correctly. **PASSES.**

### Verdict: FIXED — Closed

---

## Bug #5483 — Generator Tools Mix CustomEvent with LLM Tool Output

| Field | Value |
|-------|-------|
| **Created** | ~2025-11 |
| **Labels** | `bug` |
| **Fix PR** | #6146 (partial) |

### Problem

`CustomEvent` instances yielded from tool generators are concatenated into `function_call_output` (the string sent to the LLM), polluting model input with UI-only metadata.

### Code Trace

**Fixed path** — `agent/_tools.py:659-661`: In the `continue_run` tool execution path, `CustomEvent` is yielded as a separate event WITHOUT being added to `function_call_output`.

**Still broken** — `models/base.py` lines 2018, 2473, 2605: In the main `run_function_call` paths (sync, async, streaming), `function_call_output += str(item)` still concatenates `CustomEvent` into LLM-facing output.

### Verdict: PARTIAL — Remains Open

---

## Bug #5466 — AGUI Event Order Error (STEP_FINISHED after TEXT_MESSAGE_START)

| Field | Value |
|-------|-------|
| **Created** | 2025-11-20 |
| **Labels** | `bug` |
| **Matched PR** | #4972 (merged 2025-10-11) |

### Assessment

Bug was filed 40 days AFTER PR #4972 was merged. The PR fixed "message after tool call errors" but the reported issue is specifically about reasoning tools causing STEP_FINISHED after TEXT_MESSAGE_START without TEXT_MESSAGE_END. Different manifestation.

### Verdict: NOT_FIXED — Remains Open

---

## Bug #5493 — MCP Reset Not Handled Gracefully

| Field | Value |
|-------|-------|
| **Created** | 2025-11-22 |
| **Labels** | `bug` |
| **Matched PR** | #5181 (merged 2025-10-28) |

### Assessment

Bug was filed 25 days AFTER PR #5181 was merged. The PR added MCP connection refresh handling, but the failure mode (NoneType error during mid-call server reset) is a different scenario — connection drops during active tool execution.

### Verdict: NOT_FIXED — Remains Open

---

## PARTIAL Bugs (Matching PRs Don't Fully Address Issue)

> These 5 bugs had PRs that partially overlapped with the reported issue,
> but code tracing confirms the core problem remains.

| Bug | Title | Matched PR | Why Partial |
|-----|-------|-----------|-------------|
| #5165 | `dimensions` lost for `text-embedding-v4` | #6143 | PR passes `dimensions` only for `text-embedding-3*` or when `base_url` is set; doesn't universally handle `text-embedding-v4` |
| #5827 | Router selector can't access run context | #5107 | PR adds `session_state` to routers, but reported failure is missing `run_context` parameter |
| #5462 | Team/tool event `run_id` issue | #4834 | PR adds `parent_run_id` propagation but doesn't guarantee `run_id` on all event chunks |
| #5860 | `AgentOS.resync()` drops custom routers | #5354 | PR preserves docs/openapi routes but `_reprovision_routers` still rebuilds from built-ins only |
| #6533 | Custom retriever overridden by default search | #3913 | PR moves `num_documents` assignment but doesn't change retriever-vs-default selection logic |

All 5 remain OPEN.

---

## NOT_FIXED Bugs (Matching Commits Don't Address the Issue)

> These 20 bugs had keyword matches to merged commits/PRs, but PR diff review
> confirmed the fixes address different problems. All remain OPEN.

### HIGH Confidence Matches (Not Fixed)

| Bug | Title | Matched PR | Why Not Fixed |
|-----|-------|-----------|---------------|
| #4977 | FileTools encoding issue | #4521 | PR changes `knowledge/reader/*` encodings, not `tools/file.py` save path |
| #5101 | Generated media missing in `run_output` | #5127 | PR fixes input media wiring in Team, not generated output artifact population |
| #5278 | Team early stop / incomplete execution | #4743 | PR reorders Workflow early-stop bookkeeping, not team delegation logic |
| #5292 | OpenInference/LangSmith JSON-serializable error | #6318 | PR is about workflow `run_input` stringification, not tracing/export path |
| #6234 | `LearningMode.PROPOSE` no confirmation in AgentOS | #6180 | PR fixes MCP HITL flags, not learned-knowledge propose flow |
| #6301 | SSE + `header_provider` blocks concurrent MCP calls | #5988 | PR adds fail-fast validation for incompatible transport, not SSE concurrency |
| #6443 | RemoteAgent as Team member runtime error | #5987 | PR adds missing config attrs, not stream event formatting path |
| #6542 | `parse_tool_calls` shared dict refs (`[{}]*N`) | #5589 | PR changes `session_state or {}` patterns, not list-multiplication bug |
| #6094 | `header_provider` blocks parallel tool calls | #5655 | PR introduces dynamic headers, no fix for parallel-call hang |
| #6232 | `header_provider` + concurrent tools cancel-scope | #5655 | Same PR, no fix for cross-task cancel-scope teardown |

### MEDIUM Confidence Matches (Not Fixed)

| Bug | Title | Matched PR | Why Not Fixed |
|-----|-------|-----------|---------------|
| #4857 | Gemini video file upload fails | #5857 | PR changes `message.files` handling, not video code path |
| #5011 | Session summary exceeds model max length | #5394 | PR fixes async session handling, no max-length control added |
| #5157 | AsyncPostgres session history fails | #4661 | PR adds user-specific filtering, not `RunInput.from_dict` fix |
| #5453 | `respond_directly=True` blocks streaming | #4064 | PR is about `show_members_responses`, not `respond_directly` logic |
| #5948 | `reasoning_content` duplicated in ReasoningStep events | #4708 | PR changes yield conditions, no dedup/aggregation fix |
| #6115 | AsyncPostgres session "cannot pickle module" | #5633 | PR changes table auto-creation flags, not session serialization path |
| #6173 | AGUI custom events break tool behavior | #4636 | PR focuses tool-call ordering, not custom-event persistence/UUID serialization |
| #6257 | `_get_table()` cache bug causing connection explosion | #5600 | PR propagates `create_table_if_not_found`, not cache-check behavior |
| #6532 | Windows agent issue (docs redirect) | #6241 | PR adds Windows shebang parsing, but bug is indirect/external docs issue |
| #5901 | JSON parse fails with fenced code blocks | #5917 | PR fixes unclosed json blocks, not fenced blocks inside JSON string fields |

All 20 remain OPEN.

---
