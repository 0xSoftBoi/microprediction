---
name: platform-starter-kit-development
description: Workflow command scaffold for platform-starter-kit-development in microprediction.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /platform-starter-kit-development

Use this workflow when working on **platform-starter-kit-development** in `microprediction`.

## Goal

Creates runnable starter kits for new forecasting platforms, including code, requirements, and README documentation.

## Common Files

- `explorations/build_on_both/<platform>/*.py`
- `explorations/build_on_both/<platform>/requirements.txt`
- `explorations/build_on_both/README.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create a new subdirectory under explorations/build_on_both/ for the platform.
- Add main implementation file(s) (e.g., main.py, submit.py).
- Add requirements.txt for dependencies.
- Update or create README.md with platform comparison, instructions, and methodology.
- Verify code runs end-to-end or compiles.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.