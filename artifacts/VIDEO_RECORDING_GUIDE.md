# ContextForge Demo Video Guide

Target duration: 60 seconds

Official competition Space:

```text
https://huggingface.co/spaces/build-small-hackathon/ContextForge
```

Direct app URL:

```text
https://build-small-hackathon-contextforge.hf.space/
```

## Demo Input

```text
Project idea: I want to build a Gradio app that helps students prepare oral exams from a syllabus.
Target user: University students
Build target: A build-ready prompt pack for Codex
Topology: Single Prompt
Risk level: Medium
Output language: English
```

## Screenshot Kit

```text
artifacts/01-space-loaded.png
artifacts/02-fast-compile-filled.png
artifacts/03-full-control-open.png
artifacts/04-prompt-pack.png
artifacts/05-runtime-details.png
```

## Narration

```text
This is ContextForge: from fuzzy brief to build-ready agent blueprint.
Fast Compile lets a builder start with only the essential brief.
Full Control adds deeper context, contracts, failure modes, and verification criteria.
ContextForge does not generate one generic prompt.
It compiles the brief through seven isolated stages into an executable prompt architecture.
The Prompt Pack includes roles, cognitive modules, actions, output contracts, QA, and recovery.
Runtime Details shows the source, fallback reason, and duration for every stage.
Built for real builders using Codex and other AI coding agents.
```

## Video Timeline

| timestamp | action | visual proof |
|---|---|---|
| 0:00-0:07 | Open ContextForge | Hero, tagline, and seven-stage pipeline strip |
| 0:07-0:17 | Fill Fast Compile | Essential builder brief and selected topology |
| 0:17-0:27 | Show Full Control | Advanced context and contract controls |
| 0:27-0:42 | Show compiled Prompt Pack | Executable tagged prompt architecture |
| 0:42-0:54 | Show Runtime Details | Seven stage-level source/fallback/duration rows |
| 0:54-1:00 | Close on product identity | Public Space and compiler positioning |

## Build Command

```powershell
python artifacts/build_demo_video.py
```

## QA Notes

- Browser viewport used for screenshots: `1280x720`.
- The result screenshots use the real Gradio app with deterministic stage fallback for repeatable recording.
- The hosted Space separately supports seven real small-model calls when `CONTEXTFORGE_ENABLE_MODEL=1`.
- No secrets, cookies, tokens, `.env`, or private account data are visible.
- The video has no audio track, so narration can be recorded separately or used as subtitles.
