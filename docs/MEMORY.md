# Project Memory (`MEMORY.md`)

Persistent architecture decisions, verified environment facts, invariants, and active backlog for **AmbientSound**.

---

## 1. Confirmed Decisions
- **Date:** 2026-11-21 (`v0.5.2` hardened diagnostics, gain-normalized bird retention, parent-offset event reporting, single-pass WAV master export + heartbeats, `wait_for_job`, `distinct_regions`; building on `v0.5.0`/`v0.5.1`)
- **Repository:** `Ashfaak/AmbientSound` (Public GitHub repo, `main` branch, `git@github.com:Ashfaak/AmbientSound.git`).
- **License:** `Apache-2.0` (`LICENSE` + `NOTICE`). Strictly permissive runtime/dev dependencies only (no GPL/AGPL Python imports; zero extra package dependencies added in v0.5.0/v0.5.1).
- **Clean Section Compilation & Unified Output Layout (`v0.5.1`):** `compile_clean_sections` (sync and async via `start_compile_clean_sections_job`) and `process_sequence(..., output_dir=..., compile_clean_targets=...)` consolidate overall per-file masters, optional full combined exports, `<output_dir>/clean_sections/*` ASMR cuts, and `<output_dir>/clean_sections_manifest.json` in one validated directory outside `workspace/sessions`, auto-switching WAV containers to `RF64` above `3.8 GB`.
- **Shared Engine Module (`engines.py`, `v0.5.0`):** Pure-NumPy/SciPy DSP & detector primitives live in `src/ambient_sound/engines.py` so `analysis.py`, `dsp.py`, and `session.py` share identical spatial layout heuristics, FM chirp bird-protection masks (`compute_bird_protect_mask`), comb tracking, AR gap repair, excess-capped STFT masking, pleasantness/NDSI scoring, and do-no-harm guardrails (`assess_do_no_harm`, `compute_comparison_guardrails`).
- **Content-Defined Bird Protection (`v0.5.0`):** `protect_birds=True` never uses a brittle `> 3.5 kHz` frequency cutoff; instead it detects FM-modulated chirp ridges across `300 Hz – 11.5 kHz` while rejecting zero-drift synthetic piezo beeps via sub-bin parabolic peak invariance.
- **Persistent Async Job Manager (`jobs.py`, `v0.5.0`):** Long `process_sequence` runs execute on daemon threads coordinated by per-session `threading.RLock` locks and atomic `session.json` writes, persisted under `<workspace>/jobs/job_*.json` with cooperative cancellation and server-restart reconciliation.
- **Slew-Normalized Seam Step Gate (`v0.5.0`):** `verify_seams` evaluates sample discontinuities via `sample_step_excess_db` (default `<= 18.0 dB` over 2 s RMS sample diff when louder than `-30 dBFS`) alongside LUFS, 2 s RMS, and Welch spectral diffs.
- **Zero-Copy Reference Mode (`v0.4.0`):** `create_session(..., copy_source=False)` links allowed WAV sources in-place (saving 10 GB+ on multi-file workflows; `delete_take("take_000")` stays blocked and `delete_session` removes only the session directory), while `process_sequence(..., discard_take_audio_after_export=True)` and `export_take(..., discard_take_audio=True)` unlink duplicate internal `audio.wav` files after verified WAV export.
- **Streaming Architecture:** Recordings `> 120.0 s` (`STREAMING_THRESHOLD_S`) automatically execute block-wise on disk (`PCM_24` takes, zero full-file 2.76 GB arrays or duplicate `residual.wav`, including bounded-window `_stream_spectral_match_suppress`), with dynamic loudness-matched sliced residuals and Windowed Audition Mode (`<= 90 s` slices) in the Web UI.
- **Compact Event & Metric Transport (`v0.4.0`):** Full detected event lists persist in `takes/take_NNN/events.json`; `session.json` caps inline `take.events` (`<= 200`), MCP tools default to `max_events=50`, and `compact_metrics=True` strips verbose `short_term_lufs_timeline` points for long sessions.
- **Timeline-Aware & Candidate-First A/B (`v0.3.0`):** `TakeRecord` tracks `source_offset_s` and `timeline_modified`. `create_ab_test` defaults `original_take_id` to the earliest ancestor matching both candidates' timeline and matches candidate loudness (`match_original=False` default) so `-14 LUFS` masters are never pulled down by quiet `-49 LUFS` originals.
- **Shared Default Workspace (`v0.2.1`):** Default `workspace_dir` is `~/.ambientsound` (override via `AMBIENTSOUND_WORKSPACE` or `ambient-sound-ui --workspace`), ensuring MCP servers and the Web UI share the same session store across working directories.
- **UI Stack:** Zero-build FastAPI + Web Audio API dark studio served on `127.0.0.1:8765`.
- **Clean-Room Policy:** Skip reverse engineering. Keep any external black-box feature checklist strictly inside gitignored `.local/`. Never mention proprietary audio editors or vendors in tracked files or commits.

## 2. Verified Environment Facts (Local Dev Machine)
- **OS:** Windows (`x86_64`)
- **Python / Package Manager:** `uv 0.13.5` managing Python `3.12` venv (system also has Python `3.14.3`).
- **Media Binary:** `ffmpeg 8.1` + `ffprobe` available on `PATH`.
- **GPU:** Intel Iris Xe integrated GPU (no CUDA/NVIDIA GPU) — all v0.1/v0.2 workflows are **CPU-first**.
- **GitHub CLI (`gh`):** Not installed; Git SSH key (`id_ed25519`) is authenticated as `Ashfaak`.

## 3. Core Engineering Invariants
1. Audio tensors in memory are always 2D `np.ndarray` (`(num_samples, num_channels)`), `float64` internally.
2. Takes are immutable (`take_000` is never modified or individually deleted); short processed takes emit `audio.wav` (`FLOAT`), `residual.wav` (when length matches parent), `events.json`, `recipe.json`, `spectrogram.png`, and `diff_spectrogram.png`; long takes (`> 120 s`) stream `PCM_24` WAVs and compute sliced loudness-matched residuals dynamically.
3. Inline `take.events` in `session.json` is capped at `<= 200` (full event list stored in `events.json`), while `take.event_counts` summarizes all kinds and MCP tools default to `max_events=50` (non-click events first).
4. `mcp_server.py` never writes to `sys.stdout` (logs go only to `sys.stderr`).
5. All numeric fields serialized to JSON pass finite checks (`None` instead of `NaN`/`±inf`).
6. A/B loudness offsets are strictly `<= 0.0 dB` (attenuation-only, candidate-first by default with `match_original=False`).
7. User-supplied MCP paths are validated against `AMBIENTSOUND_ALLOWED_ROOTS`.

## 4. Active Backlog
- [ ] Benchmark streaming pipeline directly on multi-hour 48 kHz field recordings (`ZOOM0001.WAV`).
- [ ] Phase 2: Add optional CPU ONNX sound-event detectors (`Silero VAD`, `YAMNet`).
- [ ] Phase 3: Add optional text-queried target sound separation (`AudioSep` / `CLAPSep`).
