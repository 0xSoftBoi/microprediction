---
name: platform-starter-kit-extension
description: Workflow command scaffold for platform-starter-kit-extension in microprediction.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /platform-starter-kit-extension

Use this workflow when working on **platform-starter-kit-extension** in `microprediction`.

## Goal

Extends existing platform starter kits with new features, models, or validation utilities.

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

- Edit or upgrade main implementation file(s) in the platform directory.
- Add new utility or helper files (e.g., streaming_attacker.py, cv.py).
- Update requirements.txt if new dependencies are introduced.
- Update README.md to document new features or changes.
- Verify all files run or compile.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.