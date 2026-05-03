# DECISIONS — architectural decision records

> Lightweight ADRs. Each entry: context (what we knew at the time), decision
> (what we chose), consequences (what we accepted, what's now closed off).
> Sorted newest-first.

---

## 0015 — Pin `[tracing]` extra to langfuse v2 (LiteLLM compatibility)

**2026-05-03** · Status: accepted

**Context.** First production deploy of the chat agent (#13) crashed
LiteLLM's own Langfuse callback at boot:

```
Langfuse.__init__() got an unexpected keyword argument 'sdk_integration'
```

LiteLLM's `proxy-runtime` extra hard-pins `langfuse==2.59.7` and constructs
`Langfuse(sdk_integration=...)`. v3 dropped that kwarg without a
deprecation. Our `[tracing]` extra requested `langfuse>=3.0,<4`, pip
resolved to 3.14.6, and LiteLLM's callback registration broke. The error
is non-blocking (the request still flows) but it killed Langfuse tracing
on the gateway entirely — both LiteLLM's coarse traces AND our
fine-grained ones.

Two fixes were on the table:

- **Pin our extra to v2**: LiteLLM's callback works again. Our v3-only
  `_tracing.py` (uses `from langfuse import get_client` +
  `start_as_current_span`) gracefully fails the import on v2 and falls
  back to no-op spans. Trade: no fine-grained sub-spans from us; only
  LiteLLM's per-request traces emit.
- **Make `_tracing.py` support both v2 and v3 APIs**: full
  observability under either pin. Cost: write a v2 adapter
  (`Langfuse() + trace().span()` shape, very different from v3's OTel-
  flavoured `start_as_current_span` context manager).

**Decision.** Pin to v2 immediately; defer the v2 adapter.

The fast path was the right call because (a) LiteLLM's callback was
the production-blocking failure, (b) LiteLLM's per-request traces
already give the most-asked-for observability ("which model, what
input, what output, latency"), (c) the v2 adapter is real engineering
work that shouldn't gate a hotfix.

**Consequences.**
- ✅ LiteLLM Langfuse callback resumes immediately on next deploy
- ✅ Our `_tracing.py` no-ops cleanly on v2 (verified locally) — no
  crashes, no silent corruption
- ❌ Our fine-grained sub-spans (workflow_stem, prompt_id,
  comfyui.wait_for_completion latency, agent iteration tree) are
  invisible until a v2 adapter lands. The agent's "which tool got
  called" stays observable via LiteLLM's request/response capture.
- ❌ When LiteLLM eventually upgrades to langfuse v3, this pin
  becomes a problem in reverse. Treat it as a known migration debt
  (track in CHANGELOG).

---

## 0014 — Agent must use AsyncOpenAI inside an async LiteLLM proxy

**2026-05-03** · Status: accepted

**Context.** First deploy of #13 hung for 2+ minutes per chat turn,
no error, no progress. Diagnosis took longer than the fix because
the proxy logs showed nothing useful — the request just sat there.

Root cause: the agent's LLM call used the synchronous `OpenAI(...).
chat.completions.create(...)`, which inside an async coroutine blocks
the entire event loop. The agent's `PROTOBANANA_AGENT_BASE` points
at the same gateway (`http://localhost:4000/v1`), so the agent calls
back through LiteLLM. But the sync call held the event loop while
waiting on a response that needed the same loop to be served.
Instant deadlock. The block was invisible: no exception, no "request
in flight" log line, just an infinitely waiting socket.

**Decision.** Hard rule: **any LLM client invoked inside protoBanana's
async path uses `AsyncOpenAI` and is awaited.** The agent module
already had `async def run`; the OpenAI client was the one
synchronous call. Fix was a 2-line change.

Linting/contract: `protobanana.agent._build_lm_client` returns
`AsyncOpenAI`, and tests use `AsyncMock` for the `chat.completions.
create` mock so an accidental switch back to sync would fail the
suite.

**Consequences.**
- ✅ Chat turn that previously hung forever now completes in seconds
- ✅ Tests catch a regression to sync (AsyncMock vs MagicMock — sync
  call on AsyncMock returns a coroutine that's never awaited, fail
  pattern is loud)
- ❌ Future contributors who reflexively reach for `OpenAI()` will
  hit the same trap. Documented in `agent.py`'s
  `_build_lm_client()` docstring + this ADR.
- This trap generalises: any sync HTTP/LLM call inside the proxy's
  async path will deadlock the same way if it loops back. Worth
  remembering when writing future tools that call out.

---

## 0013 — Tool-use chat agent on `/v1/chat/completions`

**2026-05-03** · Status: accepted

**Context.** The Phase 1 chat path was a deterministic
keyword→workflow dispatcher: walk message history, extract latest
user prompt, classify the operation by keyword match, run one image
op, embed the result as a markdown data URL. That worked for simple
imperative prompts (`draw a cat`, `make her shirt blue`) but
couldn't handle the things users actually do in conversation:

- Conversational replies — `thanks!` got an image of "thanks!"
  generated, because the keyword router can't *not* run an op
- Clarifying questions — `make it red` was ambiguous (the *what*?)
  but had no way to ask
- Chained operations — `remove the bg, then add a sunset` needed
  two ops in one reply; the keyword router picked one
- Natural feedback — `actually I prefer the previous one` had
  nowhere to land; the router had no concept of "no op needed"

These weren't fixable with smarter keywords. The chat path needed
an LLM that owned the entire conversation surface; image ops became
tools it could choose to call.

**Decision.** Replace the keyword classifier with a tool-use agent
loop on the chat path:

1. Build OpenAI-shape tool definitions for all 6 image ops
   (`generate_image`, `edit_image`, `region_edit`, `remove_background`,
   `multi_ref_compose`, `outpaint`).
2. On each chat completion, call the configured LM (default
   `protolabs/fast` = Qwen3.6-35B-A3B-FP8) with the chat history +
   tool list.
3. If the LM returns text only → return it (conversational reply).
   If the LM returns tool calls → execute each against ComfyUI →
   feed `{success, image_size_bytes}` back to the LM → loop. Cap
   iterations at 3.
4. The LM never sees image bytes. Server-side state holds the most
   recent image + the full image list for `multi_ref_compose`. The
   final response embeds the last produced image as a markdown
   data URL (same shape as before, so existing clients keep
   rendering).
5. Keep the keyword classifier as a fallback. When the agent is
   disabled (no `PROTOBANANA_AGENT_BASE`), the openai client isn't
   importable, or the first LM call fails, the provider falls
   through to the existing keyword dispatch — degraded UX
   (image-only output) but the system stays up.

**Consequences.**
- ✅ Conversational replies, clarifying questions, multi-step
  chains all become possible
- ✅ The natural way to add Phase 8+ ops is "add a new tool
  definition + executor" — no new keyword arms, no new dispatch
  branches
- ✅ Smaller routing models (Qwen-fast) work but need anchored
  prompts; larger models (Claude, GPT-4) would need less
  hand-holding. `PROTOBANANA_AGENT_MODEL=protolabs/smart` switches
  to a thinking model when routing quality matters more than
  latency.
- ❌ +500-800 ms latency per chat turn (the LM call before any
  image work starts)
- ❌ Failure surface expanded: prompt drift, hallucinated tool
  args, infinite tool loops. Mitigated by `temperature=0`,
  `max_iterations=3`, and the keyword fallback.
- ❌ Tool descriptions become a *contract* the LLM reads; bad
  copy → bad routing. Locked the description style in
  `protobanana/tools.py`.
- See ADR 0014 (AsyncOpenAI) and 0015 (langfuse v2 pin) for the
  production-deploy bugs this surfaced.

---

## 0012 — Static workflow validator + e2e smoke as the pre-merge gate

**2026-05-03** · Status: accepted

**Context.** Three predecessor incidents took the gateway alias offline:

1. `_meta` top-level key crashed ComfyUI as an orphan node (PR #50)
2. `ImageScaleToTotalPixels` silently added `resolution_steps` as required
3. `edit_qwen_image_2511.json` shipped with `CLIPTextEncode` instead of
   `TextEncodeQwenImageEditPlus` — the workflow was syntactically valid,
   submitted cleanly, returned an image; the image just had nothing to
   do with the input (the conditioning bug, see ADR 0011)

The first two are pure schema mismatches. The third is a semantic bug —
the workflow is structurally fine, but the *meaning* is wrong: text-only
conditioning routed to an instruction-edit model. Static validation can
never catch the third.

**Decision.** Two-tier pre-merge gate:

- **Static validator** (`scripts/validate_workflows.py`) hits ComfyUI's
  `/object_info` and asserts: every node has a real `class_type`, every
  required input is present, every COMBO value is in the allowed list.
  Runtime-substituted fields (e.g. `LoadImage.image` placeholders) are
  whitelisted via `_RUNTIME_SUBSTITUTED`. Catches schema drift.

- **E2E smoke** (one-off submission with a known-recognizable input,
  visual or numeric assertion on the output). For edit-shaped workflows:
  red square + white circle + prompt that should preserve the background.
  Output's average RGB confirms input was respected. Catches semantic bugs.

Static is fast (<1s) and runs in pytest with `COMFYUI_BASE_URL`. E2E is
slower (~20s per submission) and lives as a documented script + checklist
item in DECISIONS / `validating-workflows.md`, not blocking CI yet.

**Consequences.**
- ✅ Schema-class bugs blocked at PR-time (caught the BiRefNet `class_type`
  bug on first run)
- ✅ Semantic bugs caught before user reports (would have caught the
  conditioning bug if the e2e step had existed)
- ✅ Documents the gap explicitly so we don't pretend the validator is
  enough
- ❌ E2E smoke isn't automated yet — relies on the dev to run it. Worth
  promoting to CI once we have a managed ComfyUI test endpoint.
- ❌ Validator depends on a live ComfyUI; can't run in pure-unit CI.
  Mitigated by graceful skip when `COMFYUI_BASE_URL` is unset.

---

## 0011 — `TextEncodeQwenImageEditPlus` is the only correct conditioning path for edit/multiref

**2026-05-03** · Status: accepted

**Context.** Initial `edit_qwen_image_2511.json` and
`multiref_qwen_image_2511.json` used the generic SDXL-era pattern:

```
LoadImage → ImageScale → VAEEncode → KSampler(latent_image)
CLIPTextEncode("the prompt") → KSampler(positive)
CLIPTextEncode("negative")  → KSampler(negative)
```

This is correct for SDXL/SD3-style img2img where the input image acts as
a noisy starting point and `denoise < 1.0` preserves some of it. It is
**wrong** for Qwen-Image-Edit, where:

- The model expects the image as **visual conditioning** (encoded via
  the Qwen2.5-VL vision tower, attended to during denoising), not as a
  noisy latent.
- The published example workflows always run `denoise=1.0` — the
  `latent_image` provides spatial dimensions, but the actual edit
  signal comes from the text-encoder branch with the image attached.

With our broken pattern: at `denoise=1.0` the latent_image is fully
re-noised, and `CLIPTextEncode` only sees text. Net result: zero image
conditioning → fresh unrelated output → user-visible "edit produced a
new image."

**Decision.** Both edit and multiref use `TextEncodeQwenImageEditPlus`
on positive AND negative encoders, with the scaled input image piped
into `image1` (and `image2`/`image3` for multiref). The negative
encoder receives the same image so the model has consistent visual
context across both conditioning streams.

**Consequences.**
- ✅ Edit actually edits — input image is respected, output is a
  modified version
- ✅ Same encoder shape works for 1-3 images via image1/2/3 — keeps
  edit and multiref symmetric
- ✅ Documents that "looks like img2img" is a misleading shape for
  instruction-edit models — important for any future edit-class model
  we adopt
- ❌ `_set_prompt()` helper introduces a small branch (write to
  `prompt` for Qwen edit encoders, `text` for legacy CLIPTextEncode).
  Acceptable: workflows that haven't migrated keep working.
- ❌ The bug shipped to production for ~2 days. Caught by user
  testing in the Gradio app, not by tests. Drove ADR 0012.

---

## 0010 — Standalone repo extraction (`protoBanana`)

**2026-05-03** · Status: accepted

**Context.** Provider lived inline in `homelab-iac/stacks/ai/config/litellm/
providers/comfyui_image.py` after PRs #52-#53. As Phases 2-7 came into
scope, the inline approach was hard to test, hard to share, hard to publish,
and hard to track in commits.

**Decision.** Extract to `protoLabsAI/protoBanana`. Apache-2.0.
pip-installable. Workflows ship with the package. Tests are first-class.
Documentation is the publishable artifact.

**Consequences.**
- ✅ Test suite is now isolated; can run in CI on every PR
- ✅ External users can install without joining the protoLabs repo
- ✅ Phases 4-7 land as PRs against this repo, not infra commits
- ❌ One more repo to maintain and version
- ❌ homelab-iac must now `pip install protobanana` from a tag, not
  read from a mounted directory — adds release discipline

---

## 0009 — Operation enum, single classifier function

**2026-05-03** · Status: accepted

**Context.** Need to choose between (a) one big classifier with a switch
statement, (b) a chain-of-classifiers (each operation has its own bool
predicate), (c) a state machine.

**Decision.** Single `classify_operation()` returning an `Operation`
enum. Inside it, a fixed priority order: explicit-mask > bgremove >
outpaint > inpaint > region_edit > multiref > edit > gen.

**Consequences.**
- ✅ Adding a new operation = add to enum + add a check + add tests. ~30 LOC.
- ✅ Priority is auditable — read top-to-bottom in the function
- ❌ Tightly coupled — every new op requires editing the central function
- ❌ Order matters; reordering an arm can break previously-passing tests

---

## 0008 — Markdown-embedded data URL for chat output

**2026-05-03** · Status: accepted

**Context.** `/v1/chat/completions` with image output has two protocols:

1. Multimodal `content` list with `image_url` parts (OpenAI's vision
   protocol; well-defined but client-side rendering is patchy)
2. String `content` with markdown image embed `![alt](data:...)` (renders
   in any markdown-capable UI; not OpenAI-canonical)

**Decision.** Use option 2 (markdown embed) by default. Phase 7 may add
a config flag to switch per `model_list` entry.

**Consequences.**
- ✅ Works in Open WebUI, Slack rich-text bridges, Discord, GitHub
  comments, etc. — any markdown renderer
- ✅ Client SDKs that just print the response.content show a working data
  URL that browsers can open
- ❌ Programmatic detection harder — clients must regex-extract
- ❌ Token-count accounting weird — base64 is huge in token terms but
  zero useful signal
- ❌ Diverges from OpenAI's preferred multimodal output protocol

---

## 0007 — Cap reference images at 3

**2026-05-03** · Status: accepted (Qwen ceiling)

**Context.** Qwen-Image-Edit-2511 specs cleanly handle up to 3 reference
images via parallel encode + conditioning stack. Nano-Banana 2 supports
14. Some clients might pass more.

**Decision.** Hard cap at 3 in the provider. Take first-3-in-document-order
when more arrive. Document the limit in HOWTO + PHASES.

**Consequences.**
- ✅ Predictable behavior; no silent degradation past N=3
- ✅ Matches model spec — pushing past 3 hurts quality even if the
  workflow tolerates it
- ❌ Can't compete with nano-banana on multi-ref-heavy tasks until Qwen
  ships a higher-ref variant
- ❌ Open question: should we eventually pre-mux 4+ images into 3 via
  semantic selection? Defer until real demand.

---

## 0006 — Server-side workflow substitution (not client)

**2026-05-03** · Status: accepted (PR #52 outcome)

**Context.** Open WebUI's `IMAGE_GENERATION_ENGINE=comfyui` mode requires
the client to know ComfyUI's prompt schema and node-IDs. PR #49-#51
showed the contract is brittle: each Open WebUI version shifts the env
schema (`COMFYUI_WORKFLOW_NODES` went from CSV string → JSON list of
strings → list of arrays).

**Decision.** Provider mutates node IDs server-side in Python. Clients
just send standard OpenAI requests; they need to know nothing about
ComfyUI.

**Consequences.**
- ✅ Open WebUI version drift doesn't affect us
- ✅ Adding a client (protoCLI, raw curl, etc.) is zero work
- ✅ Workflow JSONs can be debugged standalone in ComfyUI's UI
- ❌ Provider must know each workflow's node-ID conventions
  (mitigated by per-route `substitute()` functions that own the convention)
- ❌ If we want to support arbitrary user-supplied workflows, we'd need
  a node-mapping config — defer until that's a real ask

---

## 0005 — BiRefNet default, RMBG-2.0 opt-in

**2026-05-03** · Status: accepted

**Context.** RMBG-2.0 outperforms BiRefNet (90% vs 85% on bg-removal
benchmark) but is **CC BY-NC 4.0** — non-commercial only. BiRefNet is
open-license, commercial-safe.

**Decision.** Default workflow `bgremove_birefnet.json` ships with
BiRefNet. `bgremove_rmbg2.json` is the opt-in alternative for personal
or non-commercial users. Both workflows live in `workflows/`; the user
picks via `model_name` (`protolabs/qwen-image-bgremove` or
`protolabs/qwen-image-bgremove-rmbg`).

**Consequences.**
- ✅ Out-of-box installs are commercial-safe
- ✅ Users who don't care about license get the higher-quality option
  by deliberate choice
- ✅ Documentation can be honest about the trade
- ❌ Two workflows to maintain (mitigated: bg-remove workflows are
  trivial — 3 nodes each)
- ❌ Slightly more complex model_list entry naming

---

## 0004 — `huggingface-cli download` cache-only by default

**2026-05-03** · Status: accepted (after the deletion incident)

**Context.** Used `--local-dir` for a "redundant" download, then `rm -rf`'d
the duplicate to free 29 GB. **Six ComfyUI symlinks broke** because
`--local-dir` shares blob inodes with the cache via hardlinks; deleting
the local-dir cascaded into the cache.

**Decision.** Default to cache-only downloads (`HF_HOME=/mnt/models/
huggingface huggingface-cli download <repo> <files>`, no `--local-dir`).
Document in lab CLAUDE.md storage rules. If `--local-dir` is required,
also pass `--local-dir-use-symlinks False` so files are full copies.

**Consequences.**
- ✅ No more cascading deletes
- ❌ Slightly less convenient for "I want files in a specific location"
  use cases — must accept the cache layout

---

## 0003 — Strip metadata keys (`_meta`, `_doc`) from workflow JSON

**2026-05-03** · Status: accepted (after PR #50)

**Context.** Wanted documentation embedded in workflow JSONs (model
filenames, expected paths, license notes). Initially used a top-level
`_meta` key.

ComfyUI iterates **all top-level keys** as node IDs and validates each
has `class_type`. `_meta` raised `missing_node_type` → HTTP 500 → Open
WebUI failed parsing the response with `Expecting value: line 1 column 1
(char 0)`.

**Decision.** Move documentation to `workflows/README.md` and per-workflow
`_doc` fields that the loader strips before submission. The loader filters
top-level keys, keeping only those with `class_type` set.

**Consequences.**
- ✅ Workflows are valid ComfyUI input
- ✅ Inline `_doc` field still works for IDE tooltips and side-by-side
  reading; just doesn't get submitted
- ❌ One more thing to remember when authoring workflows; tested in
  `tests/test_workflow_loader.py`

---

## 0002 — Gateway-routed image gen, not Open WebUI direct

**2026-05-03** · Status: accepted (PR #52)

**Context.** Open WebUI's `IMAGE_GENERATION_ENGINE=comfyui` was failing
in three ways: brittle workflow contract, no intent inference (required
explicit UI gesture), single-client coupled (couldn't reuse for protoCLI).

**Decision.** Build a LiteLLM CustomLLM provider that exposes ComfyUI as
an OpenAI-compatible image endpoint. Open WebUI swaps to
`IMAGE_GENERATION_ENGINE=openai` pointing at the gateway.

**Consequences.**
- ✅ Same call shape as `protolabs/nano-banana-2`. Architectural symmetry.
- ✅ Every client gets image gen for free.
- ✅ Langfuse traces, Prometheus metrics, retries — gateway features
  apply automatically.
- ❌ One more service in the chain (gateway → provider → ComfyUI vs
  Open WebUI → ComfyUI directly). Adds ~20-50ms.

---

## 0001 — Qwen-Image-2512 + Qwen-Image-Edit-2511 over alternatives

**2026-05-02** · Status: accepted

**Context.** OSS image gen + edit landscape (May 2026):
- HunyuanImage 3.0 Instruct: 80B MoE, top Elo, but too big for our 96 GB
- Qwen-Image-2512: 20B MMDiT, #1 AA Arena both gen+edit, native 2K
- BAGEL-7B-MoT: 7B active, has world-modeling, smaller
- FLUX.1 Kontext: 7s/edit, weak text rendering

**Decision.** Qwen-Image-2512 (gen) + Qwen-Image-Edit-2511 (edit).
Already on disk via lab `experiments/image-gen-eval`. Best fit for our
GPU. Strongest text rendering of the OSS options.

**Consequences.**
- ✅ Best OSS quality at our scale
- ✅ Native multi-ref (3-image cap) baked into Edit-2511
- ❌ 3-ref ceiling vs nano-banana's 14
- ❌ Two UNets (gen + edit) need swap-in/swap-out
- ❌ Quality 6-12 months behind frontier closed models on hardest cases
