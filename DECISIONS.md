# DECISIONS — architectural decision records

> Lightweight ADRs. Each entry: context (what we knew at the time), decision
> (what we chose), consequences (what we accepted, what's now closed off).
> Sorted newest-first.

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
