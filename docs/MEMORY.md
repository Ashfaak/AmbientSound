# Project Memory (`MEMORY.md`)

Persistent architecture decisions, verified environment facts, invariants, and active backlog for **AmbientSound**.

---

## 1. Confirmed Decisions
- **Date:** 2026-11-20
- **Repository:** `Ashfaak/AmbientSound` (Public GitHub repo, `main` branch, `git@github.com:Ashfaak/AmbientSound.git`).
- **License:** `Apache-2.0` (`LICENSE` + `NOTICE`). Strictly permissive runtime/dev dependencies only (no GPL/AGPL Python imports).
- **Primary Stack:** Python 3.12 (`uv` environment), `mcp>=2.0.0` (`MCPServer`, `stdio`), `numpy`, `scipy`, `soundfile`, `pyloudnorm`, `matplotlib`, `fastapi`, `uvicorn`, `pydantic`.
- **UI Stack:** Zero-build FastAPI + Web Audio API dark studio served on `127.0.0.1:8765`.
- **Clean-Room Policy:** Skip reverse engineering. Keep any external black-box feature checklist strictly inside gitignored `.local/`. Never mention proprietary audio editors or vendors in tracked files or commits.

## 2. Verified Environment Facts (Local Dev Machine)
- **OS:** Windows (`x86_64`)
- **Python / Package Manager:** `uv 0.13.5` managing Python `3.12` venv (system also has Python `3.14.3`).
- **Media Binary:** `ffmpeg 8.1` available on `PATH`.
- **GPU:** Intel Iris Xe integrated GPU (no CUDA/NVIDIA GPU) — all v0.1 workflows are **CPU-first**.
- **GitHub CLI (`gh`):** Not installed; Git SSH key (`id_ed25519`) is authenticated as `Ashfaak`.

## 3. Core Engineering Invariants
1. Audio tensors are always 2D `np.ndarray` (`(num_samples, num_channels)`), `float64` internally.
2. Takes are immutable (`take_000` is never modified); processed takes emit `audio.wav` (32-bit float), `residual.wav`, `recipe.json`, `spectrogram.png`, and `diff_spectrogram.png`.
3. `mcp_server.py` never writes to `sys.stdout` (logs go only to `sys.stderr`).
4. All numeric fields serialized to JSON pass finite checks (`None` instead of `NaN`/`±inf`).
5. A/B loudness offsets are strictly `<= 0.0 dB` (attenuation-only).
6. User-supplied MCP paths are validated against `AMBIENTSOUND_ALLOWED_ROOTS`.

## 4. Active Backlog
- [ ] Push initial commit to `Ashfaak/AmbientSound` once empty GitHub repository exists.
- [ ] Benchmark `ambience_patch` and `band_attenuate` on real multi-minute rain/café/forest recordings.
- [ ] Phase 2: Add optional CPU ONNX sound-event detectors (`Silero VAD`, `YAMNet`).
- [ ] Phase 4: Add chunked overlap-add streaming for long (>15 min) recordings.
