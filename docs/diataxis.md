# Diátaxis map

This site follows the [Diátaxis framework](https://diataxis.fr): four kinds of documentation, four user intents, no overlap. The sidebar is organised by quadrant; this page is the explicit map so you can find what you need by *what you're trying to do* rather than by topic.

## The four quadrants

|  | Tutorials | How-to guides |
|---|---|---|
| **Practical** | learning-oriented | task-oriented |
|  | "I want to learn protoBanana" | "I want to do X" |
|  | hand-holding, hands-on | concise recipe, no detours |

|  | Reference | Explanation |
|---|---|---|
| **Theoretical** | information-oriented | understanding-oriented |
|  | "I need to look something up" | "I want to know why" |
|  | dry, complete, structured | discursive, opinionated |

A page is in exactly one quadrant. If you find a page that mixes them, that's a docs bug — file an issue.

## Where to find each quadrant

### 📘 Tutorials — start here

Walks you from zero to working. Read top-to-bottom, do the steps.

- [Quickstart (5 min)](guide/quickstart) — clone, install, hit the gateway, get an image back.

> **Gap:** beyond the quickstart we don't yet have tutorials for "your first chat agent" or "your first custom workflow". Both could be valuable; opening as work to do.

### 🛠 How-to guides — get a thing done

You already know what protoBanana is. You want to accomplish a specific task.

- [Install protoBanana into a gateway](installation) — the full from-scratch path
- [Operate day-2](operating) — model swaps, GPU planning, troubleshooting
- [Add a new ComfyUI workflow](workflows-cookbook) — every step from "build it in the UI" to "wire it into a route"
- [Validate workflows before shipping](validating-workflows) — the static validator + the e2e smoke pattern
- [Enable the chat agent](agent) — env vars, model choice, fallback behaviour
- [Enable Langfuse tracing](observability) — what's captured, current v2 pin trade
- [Run the Gradio test/eval UI](gradio-app) — locally + as a HuggingFace Space

### 📑 Reference — lookup

You know what to do; you need a value, a schema, or an exact name.

- [API](api) — endpoints, request/response shapes
- [Architecture](architecture) — component map, where each module lives
- [Keyword intent router](intent-router) — the fallback dispatch path's rules
- [Benchmarks](benchmarks) — quality + latency methodology + numbers

### 💡 Explanation — the why

Background reading. None of these tells you what to *do*; they tell you why protoBanana is shaped the way it is.

- [Proposal](deep-dives/proposal) — the strategic system design
- [Phases](deep-dives/phases) — what shipped per phase + the rationale
- [Journey](deep-dives/journey) — how we got here, including the production-deploy bug stories
- [Decisions (ADRs)](deep-dives/decisions) — architectural decision records, newest first
- [Changelog](deep-dives/changelog) — per-release log

## Why this matters

The thing that breaks docs sites is mixing intents. A tutorial that suddenly explains *why* the architecture is shaped a certain way slows the reader down. A reference page that breaks into a tutorial mid-table makes it impossible to skim. Diátaxis is a discipline that keeps each page in one mode.

If you're contributing docs, the quickest test before merging: **what is the reader trying to do?** If you can name an intent that doesn't fit one of the four quadrants, the page is probably trying to do too much — split it.
