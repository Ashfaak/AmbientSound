# Open-Source License Compatibility Analysis

> **Disclaimer:** This document summarizes technical license compatibility findings from repository research for engineering decision-making. It is not formal legal advice.

**AmbientSound** is licensed under **Apache-2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE)). Choosing Apache-2.0 gives users and contributors permissive open-source reuse, explicit patent protection, and clean compatibility with most Python/Rust audio tooling—provided we enforce strict boundaries on copyleft libraries and non-commercial ML weights.

---

## 1. Reference Ecosystem Evaluation (vs. AmbientSound `Apache-2.0`)

| Reference Project | Repo License | Compatibility with `Apache-2.0` | What Can Be Used / Adapted | Hard Exclusions & Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **`djynnius/ferraus`** | `Apache-2.0 OR MIT` | **Fully Compatible** | Architecture patterns (JSON-first agent tool outputs, deterministic `.json` provenance sidecars, `stdio` MCP design), or invoking the `ferraus` CLI binary optionally in future phases. | Omit optional `ferraus-engrave` (LGPL-3.0 sheet-music helper, irrelevant to ambient audio). Audit model weights downloaded separately on first run (`DeepFilterNet` MIT/Apache-2.0, `RNNoise` BSD-3-Clause, `Demucs` MIT). |
| **`GeekatplayStudio/music-suite`** | `MIT` | **Core Code Compatible** | MIT-licensed architectural patterns—especially its read-only-by-default MCP mutation gating concept and local loopback workflow design. | **Do NOT** import or bundle its optional **GPL-3.0** dependencies (`pedalboard`, `rubberband` / `pyrubberband`). **Do NOT** bundle restricted or non-commercial stem/generation weights. |
| **`psyb0t/docker-audiolla`** | `WTFPL` | **Adaptable / Relicensable** | Wrapper/API concepts can be studied or adapted into `Apache-2.0`. Crucially, adopt its explicit opt-in environment-flag guardrail for non-commercial weights (`AMBIENTSOUND_ALLOW_NONCOMMERCIAL=0` by default). | Do **not** license AmbientSound under WTFPL (not OSI-approved and lacks a warranty disclaimer). **Exclude** GPL-3.0 bundled tools (`matchering`, `pedalboard`) and CC-BY-NC / Stability Community weights (`MusicGen`, `Stable Audio Open`). |
| **`fadelabs/phantom`** | `AGPL-3.0` | **Incompatible for Code Derivation** | High-level domain concepts only (evaluating loudness, stereo phase/correlation, frequency masking, and spectral diagnostics via MCP). | **Never** copy, link, fork, or derive code from `fadelabs/phantom` into AmbientSound; doing so would require relicensing AmbientSound under `AGPL-3.0`. Users wanting Phantom alongside AmbientSound can configure it separately as an external peer MCP server. |

---

## 2. AmbientSound Dependency License Policy

### Allowed in Core Dependencies (`pyproject.toml`)
- `Apache-2.0`
- `MIT`
- `BSD-2-Clause` / `BSD-3-Clause`
- `ISC`
- `PSF-2.0` (Python Software Foundation)
- `MPL-2.0` (file-level weak copyleft; acceptable as an unmodified dependency)
- `LGPL-2.1+` / `LGPL-3.0` *only* via dynamically linked system shared libraries (e.g., `libsndfile` via `soundfile` CFFI bindings), never statically linked proprietary blobs.

### Prohibited in Core Dependencies (Enforced by `scripts/check_hygiene.py`)
- **GPL-2.0 / GPL-3.0** Python/Cython/Rust packages (`pedalboard`, `matchering`, `pyrubberband`, `essentia` unless commercial/separate process).
- **AGPL-3.0** libraries (`phantom`, etc.).
- **Non-commercial / research-only weights** (`CC-BY-NC-*`, Stability Community License, custom field-of-use weights like `SAM License`). If integrated in future phases as optional plugins, they must be downloaded by the user at runtime and blocked unless `AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1`.

---

## 3. Initial v0.1.0 Python Dependency Audit

| Package | Purpose | License | Status |
| :--- | :--- | :--- | :--- |
| `mcp` | Official Model Context Protocol Python SDK (`MCPServer`, `stdio`) | MIT | Approved |
| `numpy` | Core array & FFT numerics | BSD-3-Clause | Approved |
| `scipy` | IIR/SOS filters, STFT, Welch PSD, interpolation | BSD-3-Clause | Approved |
| `soundfile` | Audio WAV/FLAC I/O wrapping `libsndfile` | BSD-3-Clause (`libsndfile`: LGPL-2.1/3.0 dynamic) | Approved |
| `pyloudnorm` | ITU-R BS.1770-4 / EBU R128 integrated LUFS loudness | MIT | Approved |
| `matplotlib` | Headless waveform & spectrogram PNG generation | PSF-based (BSD-compatible) | Approved |
| `fastapi` | Local A/B Listening Studio REST API | MIT | Approved |
| `uvicorn` | ASGI local HTTP server | BSD-3-Clause | Approved |
| `pydantic` | Strict schema validation for recipes, metrics, verdicts | MIT | Approved |
| `pytest` (dev) | Unit & integration test runner | MIT | Approved |
| `httpx` (dev) | FastAPI `TestClient` transport | BSD-3-Clause | Approved |
| `ruff` (dev) | Linting & formatting | MIT | Approved |
| `ffmpeg` (external CLI) | Subprocess fallback for MP3/M4A/OGG decode & export | LGPL/GPL external binary invoked via OS process | Separate process boundary |
