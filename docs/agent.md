# The chat agent

protoBanana's `/v1/chat/completions` path is, by default, a tool-use **chat agent** — an LLM is the brain. It decides whether to respond conversationally, call an image tool, or chain multiple tools to satisfy a user request.

This was a deliberate architectural choice. The earlier dispatch was a deterministic keyword classifier: every chat turn ran one image op and returned the image. That worked for simple commands but couldn't handle:

- Conversational replies (`thanks!` got an image of "thanks!" generated)
- Clarifying questions (`make it red` was ambiguous — the *what*? — but had no way to ask)
- Chained operations (`remove the bg, then put a sunset behind` needed two ops in one reply)
- Natural feedback (`actually I prefer the previous one` had nowhere to land)

The agent owns the entire chat surface; image ops are just tools it can call.

## How the loop works

```
user msg
  ↓
ProtoBanana.acompletion(messages)
  ↓
build SYSTEM prompt + chat history (with image data URLs elided)
  ↓
loop (max 3 iterations by default):
  resp = LM.chat.completions.create(messages, tools=IMAGE_TOOLS)
  if resp.tool_calls:
    for each call:
      result = execute the tool against ComfyUI
      append { role: "tool", content: <{success, size}> } to messages
    continue
  else:
    return resp.content  +  markdown embed of last image (if any tool ran)
```

That `LM.chat.completions.create` call is a **loopback into the same LiteLLM gateway** (`PROTOBANANA_AGENT_BASE=http://localhost:4000/v1`): the provider runs *inside* the gateway and re-enters it to reach the routing LLM, so the agent's own calls inherit the gateway's routing, fallbacks, and observability. It must be async or it self-deadlocks under a single worker — see the [Agent feedback loop diagram](architecture.md#agent-feedback-loop) and [DECISIONS.md §0014](../DECISIONS.md#0014).

The LLM sees text only — never image bytes. Server-side state tracks "the most recent image" (and a list of all in-conversation images for `multi_ref_compose`). Tool results returned to the LLM are tiny dicts (`{"success": true, "image_size_bytes": 487123}`) so the conversation history stays cheap.

## Tools

| Tool | When the LLM should call it |
|---|---|
| `generate_image(prompt, size?)` | New image from text. No input image needed. |
| `edit_image(instruction)` | Whole-image transformation of the most recent image |
| `region_edit(region, edit_prompt)` | Change a NAMED sub-region (e.g. "the man's tie") |
| `remove_background()` | Sticker / alpha PNG of the most recent image |
| `multi_ref_compose(prompt)` | Blend 2-3 reference images |
| `outpaint(left/top/right/bottom, fill_prompt)` | Extend the canvas |

The full JSON schema for each tool lives in `protobanana/tools.py` — that's the source of truth the LLM reads at routing time.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `PROTOBANANA_AGENT_BASE` | OpenAI-compatible LLM URL | unset = agent disabled |
| `PROTOBANANA_AGENT_KEY` | API key (or whatever your LM expects) | `none` |
| `PROTOBANANA_AGENT_MODEL` | Model id at the LM endpoint | `protolabs/fast` |
| `PROTOBANANA_AGENT_MAX_ITERS` | Cap on tool-call iterations per turn | `3` |

Install with the `[agent]` extra to pull in the OpenAI client dep:

```bash
pip install 'protobanana[agent]'
```

When `PROTOBANANA_AGENT_BASE` is unset, the agent is **disabled** — the provider falls back to the deterministic keyword classifier path. The package keeps working without an LM, which matters for HF Spaces / dev / standalone use.

## Choosing a model

`protolabs/fast` is the recommended default — Qwen3.6-35B-A3B-FP8 (uncensored heretic), ~226 tok/s, ~500 ms to a tool-call decision on a typical 200-token prompt. No thinking tokens.

`protolabs/smart` (Qwen3.6-27B-FP8 thinking, ~150 tok/s with thinking) is a better choice when:

- The conversation requires planning across multiple operations (`make him taller, then put him in a winter coat`)
- Ambiguity needs careful resolution (`change his hat` when there are multiple men)
- You're willing to pay 1-2 s of thinking time for richer routing

Set `PROTOBANANA_AGENT_MODEL=protolabs/smart` to switch.

## Fallback behavior

The agent loop returns `None` from `agent.run()` in three cases:

1. `PROTOBANANA_AGENT_BASE` is unset (agent disabled)
2. The OpenAI client can't be imported (`pip install 'protobanana[agent]'` not done)
3. The very first LM call raised (network, auth, model unavailable)

The provider checks for `None` and **falls through to the keyword classifier path** that ships baked into protoBanana. The user sees a slightly degraded experience (no conversational replies, image-only output) but the system stays up.

If the LM call fails AFTER tools have already run, the agent returns the last produced image with a soft "I had a problem mid-conversation" note rather than dropping the work.

## Tracing

When [Langfuse tracing](observability) is on, every agent turn produces:

```
protobanana.acompletion              [parent — has metadata.path = "agent" | "keyword"]
└── protobanana.agent
    ├── protobanana.agent.iter_0
    │   └── protobanana.tool.<name>      (when LLM called a tool)
    ├── protobanana.agent.iter_1         (next iteration after tool result)
    │   └── protobanana.tool.<name>
    └── protobanana.agent.iter_2
```

Useful filters in the Langfuse UI:

- `metadata.path = "agent"` — only chat turns that went through the agent (filter out keyword fallbacks)
- `protobanana.agent.iter_*` — sort by duration to find slow LM calls
- `metadata.hit_max_iterations = true` — turns where the LLM got stuck and we capped
- `metadata.agent_fallback = true` — turns where the agent failed and we fell back

## Multi-step examples

The agent's value shows up most clearly on chained operations the keyword router can't express:

| User says | Agent does |
|---|---|
| `thanks!` | reply: "you're welcome — anything else?" — no tools |
| `make it red` (after a previous image) | reply: "the whole image red, or a specific part?" — no tools |
| `change his hat to red` | call `region_edit(region="his hat", edit_prompt="a red hat")` |
| `remove the bg, then make a sunset behind` | call `remove_background()`, then `outpaint(fill_prompt="sunset sky")` |
| `make her shirt blue and his tie green` | call `region_edit` twice in one turn |
| `actually use the previous one` | reply asking for clarification or recall the prior image — no tools |

These are the experiences the keyword router fundamentally couldn't ship.
