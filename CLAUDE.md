# CLAUDE.md

## Project

Autoresearch Agent Scaffold — a deterministic experiment harness where an LLM proposes small `search/replace` edits to `train.py` and the harness handles everything else (git, execution, parsing, logging, keep/revert).

## Architecture

Four files, strict separation:
- `main.py` — orchestration, setup, loop, recovery
- `state.py` — all I/O (state.json, git, files)
- `planner.py` — LLM context assembly, proposal, editing
- `results.py` — execution, parsing, logging, decision

## Rules

- **All edits are local.** Never push, deploy, or modify anything outside this repo.
- **Simplicity is king.** Write the simplest code that works. No abstractions until they're needed twice. No helpers for one-time operations. Three similar lines beat a premature abstraction.
- **Small changes only.** One hyperparameter tweak, one local architecture change, or one simplification per experiment.
- **Don't over-engineer.** No feature flags, no backwards-compat shims, no speculative error handling. If you can delete it, delete it.
- **Structured proposals.** The planner returns JSON only — no free-form text.
- **Deterministic harness.** Git, execution, parsing, logging, and keep/revert are never delegated to the LLM.
- **External memory.** Progress lives in files (`state.json`, `results.tsv`), not the context window.
- **Stay in scope.** Only read and edit files within this directory (`llm_scaffold/`). Do not touch the parent repo or sibling directories.
