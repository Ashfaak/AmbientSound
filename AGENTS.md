# Agent Working Rules (`AGENTS.md`)

Guidelines for AI coding agents working in the **AmbientSound** repository.

## 1. Documentation & Project Memory Discipline
- Whenever you add features, change MCP tool contracts, adjust DSP algorithms, or fix bugs:
  - Update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.
  - Update [docs/MEMORY.md](docs/MEMORY.md) if architectural invariants, hardware assumptions, or key decisions change.
  - Record DSP parameter experiments, benchmark observations, or edge cases in [docs/NOTES.md](docs/NOTES.md).

## 2. Licensing & Clean-Room Hygiene
- **Project license is `Apache-2.0`.**
- Never add GPL, AGPL, or non-commercial dependencies to `pyproject.toml` core dependencies (specifically avoid `pedalboard`, `matchering`, `pyrubberband`, `essentia`, etc.). See [docs/LICENSES.md](docs/LICENSES.md).
- Any future optional ML weights that carry non-commercial or restricted field-of-use licenses MUST remain gated behind `AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1` (default `0`/`False`).
- Maintain strict clean-room standards: never commit `.local/` notes or any proprietary vendor/product references to tracked files or commit messages.

## 3. Core Engineering Invariants
- **Audio array invariant:** Internal audio arrays flowing through `io_audio`, `analysis`, `dsp`, and `visuals` are always 2D NumPy floating-point arrays shaped `(num_samples, num_channels)`.
- **Non-destructive takes:** Never overwrite `take_000` or previously created takes. Every `process_take` run produces a new immutable `take_NNN` with `audio.wav`, `residual.wav`, and `recipe.json`.
- **JSON safety:** MCP tools and FastAPI JSON payloads must never emit `NaN` or `±inf`. Always sanitize floating metrics through `_finite_or_none`.
- **MCP `stdio` hygiene:** Never write logs or debug prints to `sys.stdout` inside `mcp_server.py` or modules invoked during MCP tool execution. Route all logging strictly to `sys.stderr`.
- **Path sandboxing:** All user-supplied filesystem paths entering via MCP tools must pass through `ensure_path_allowed` in `src/ambient_sound/config.py`.
- **Loudness-matched A/B safety:** A/B gain compensation must always be **attenuation-only** (`gain_db <= 0.0`) so playback never digitally clips.

## 4. Required Verification Commands
Run all checks before committing or concluding a coding task:

```bash
uv run ruff check .
uv run pytest -v
uv run python scripts/check_hygiene.py
```
