---
title: ContextForge
emoji: ⚒️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: true
---

# ContextForge / Agent Prompt Compiler

ContextForge compiles messy software, app, and agent ideas into executable prompt architectures. It is a compiler pipeline, not a generic prompt generator.

**GitHub:** https://github.com/rthgit/ContextForge

**Competition Gradio Space:** https://huggingface.co/spaces/build-small-hackathon/ContextForge

**Backup Gradio Space:** https://huggingface.co/spaces/RthItalia/ContextForge

**Demo video:** https://raw.githubusercontent.com/rthgit/ContextForge/main/artifacts/contextforge-demo.mp4

**Tagline:** From fuzzy brief to build-ready agent blueprint.

## Backyard AI Fit

- Built for real builders using AI coding agents.
- Real problem: vague briefs make Codex and other agents produce wrong code, generic UI, or incomplete workflows.
- Real use evidence: this architecture was used to coordinate Trollsona development, including UI refactor, model cascade, QA, packaging, and video automation.
- Small-model fit: ContextForge decomposes a hard prompt-writing task into seven smaller calls so a small model can handle it.

The backend always executes seven isolated modules sequentially:

1. intake analysis
2. topology decision
3. Vital Few / Vital Spot extraction
4. reasoning architecture selection
5. prompt pack generation
6. QA / repair
7. final assembly

Every module attempts its own small-model call. If one call fails, only that stage uses a deterministic fallback and the pipeline continues. Runtime Details shows the source used by every stage.

Each module also has a bounded token budget appropriate to its contract. `CONTEXTFORGE_MAX_NEW_TOKENS` is the global ceiling, while stage budgets keep the seven-call CPU path practical.

## Topologies

- Single Prompt
- Cascade
- Context Pack
- Agent Workflow

Auto topology uses Cascade when multiple expertise areas or dependent outputs are required. Agent Workflow is preferred for agentic or critical-risk work. Context Pack stabilizes incomplete briefs.

## Safety

- Private reasoning remains internal.
- Generated prompts never request full chain of thought.
- Controlled Tree of Thought exposes only `strategy | upside | risk | cost | selected`.
- Public reasoning fields are limited to decision summary, assumptions, risks, verification steps, and final answer.
- QA repairs missing tags, contracts, verification, repair logic, and unsafe reasoning requests.

## Runtime

Recommended Hugging Face Space variables:

```text
CONTEXTFORGE_ENABLE_MODEL=1
CONTEXTFORGE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
CONTEXTFORGE_MID_MODEL_ID=RthItalia/nano_compact_3b_qkvfp16
CONTEXTFORGE_HIGH_MODEL_ID=Qwen/Qwen3-32B
CONTEXTFORGE_MAX_NEW_TOKENS=1800
```

Runtime selection:

1. high model only when CUDA is available
2. compact mid model when CUDA is available
3. Qwen 0.5B on public CPU Space
4. deterministic stage-level fallback

For a fast local deterministic run:

```powershell
$env:CONTEXTFORGE_ENABLE_MODEL='0'
python app.py
```

## Local QA

```powershell
python -m py_compile app.py
python test_contextforge.py
python app.py
```

The QA script verifies all four topologies, independent stage execution, required tags, chain-of-thought safety, controlled Tree of Thought output, and stage-level fallback continuity.

## Demo Assets

- Demo video: `artifacts/contextforge-demo.mp4`
- Recording guide: `artifacts/VIDEO_RECORDING_GUIDE.md`
- Submission pack: `SUBMISSION.md`
