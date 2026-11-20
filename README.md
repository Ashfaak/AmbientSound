# AmbientSound

**MCP-driven ambient audio cleanup and loudness-matched A/B listening studio.**

AmbientSound is an open-source (`Apache-2.0`) audio restoration toolkit built specifically for **field recordings and ambient soundscapes** (rain, wind, forests, cafés, urban rooms, ocean surf). Unlike speech-only denoisers that treat ambient background texture as noise to be erased, AmbientSound treats the ambient bed as the **desired signal** and targets **localized distractions**—hum, wind rumble, clicks, stray horns, voices, or handling bumps—while keeping the surrounding soundscape natural.

An AI model interacts with AmbientSound over the **Model Context Protocol (MCP)** to inspect psychoacoustic metrics, spectrograms, and detected distraction spans, propose multiple cleanup filter chains (non-destructive *takes*), and spin up **loudness-matched A/B listening tests** in a local Web Audio studio where you choose the winner and leave feedback for the next tuning pass.

---

## Core Workflow

```mermaid
flowchart LR
    A[Raw Ambient Recording] --> B[Non-Destructive Session\ntake_000 Original]
    B --> C[Analysis + Distraction Detection\n+ Spectrogram PNG]
    C --> D[AI Agent via MCP Server\nstdio tools]
    D --> E[Candidate Take A & Take B\n+ Residual Diff WAVs]
    E --> F[Local Web A/B Studio\nLoudness-Matched + Time-Locked]
    F --> G[User Verdict, Artifact Tags\n& Timestamped Comments]
    G --> D
    E --> H[Export Winning Take\nWAV / FLAC / MP3 / OGG]
```

## Key Features (v0.2.0)

- **14 Ambient-First CPU DSP Filters (Streaming & In-Memory, Permissive Stack):**
  - `highpass_rumble`: Zero-phase Butterworth high-pass for wind/traffic rumble.
  - `dehum_notch` & `drone_notch`: Zero-phase cascaded notches for 50/60 Hz mains hum and arbitrary machine-drone frequencies (`30–1000 Hz` auto-detected).
  - `declick_impulse`: MAD spike detection + PCHIP sample interpolation for mic clicks/pops.
  - `spectral_denoise`: Linked-stereo soft STFT spectral gate/Wiener denoiser.
  - `band_attenuate`: Feathered time-frequency attenuation supporting `"swell"` envelopes, up to `5 s` fades, and batched multi-span lists (`spans`).
  - `ambience_patch`: Replaces one or many distracted spans with clean matching texture via up to `5 s` equal-power crossfades.
  - `trim_range`, `cut_regions`, & `fade_edges`: Extract ranges, cut bad stretches with equal-power crossfades, and add smooth start/end fades.
  - `parametric_eq`, `soft_compress`, `loudness_normalize`, & `true_peak_limiter`: Tone shaping (`lowshelf` warmth / `highshelf` air), gentle RMS compression, 2-pass streaming BS.1770 LUFS leveling (`-14 LUFS` YouTube/ASMR ready), and lookahead limiting.
- **Large-File Bounded-RAM Streaming (`> 120 s` / 1-Hour+ Recordings):** Streams blocks for `create_session`, `analyze_take`, `analyze_region`, `process_take`, `render_spectrogram`, and `export_take` without loading multi-GB float arrays into RAM.
- **Non-Destructive Take DAG:** Every processing pass writes an immutable `take_NNN` (`audio.wav`, `recipe.json`, `spectrogram.png`, plus `residual.wav` / dynamic sliced residuals and `diff_spectrogram.png`).
- **Bias-Free Local A/B Listening Studio (with Windowed Audition Mode):**
  - Time-locked synchronized Web Audio playback across Original, Take A, Take B, and Removed Residuals (`1`/`2`/`3`/`4` hotkeys, `12 ms` click-free crossfades).
  - Automatic `<= 90 s` Windowed Audition Mode on long recordings so multi-hour files audition effortlessly in the browser.
  - Attenuation-only LUFS loudness matching (`<= 0 dB` gain offsets), loop focus spans, Blind Mode, and structured feedback tags (`Natural ambience`, `Metallic / swishy`, `Hollow gap`, etc.).
- **Path-Sandboxed MCP Server (`stdio`, 13 Tools):** Includes `health_check`, `analyze_region`, windowed `render_spectrogram`, clean JSON responses, and strict path allowlisting.

---

## Quickstart

### Prerequisites
- Python 3.11–3.13 (project targets **Python 3.12** via `uv`)
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH` (optional for WAV/FLAC; required for MP3/M4A/OGG decode & export)

### 1. Install dependencies
```bash
uv sync --python 3.12 --all-groups
```

### 2. Generate and audition the built-in synthetic demo
```bash
uv run ambient-sound demo
uv run ambient-sound-ui
```
Open the printed URL (for example `http://127.0.0.1:8765/?session=...&ab=...`) in your browser and press:
- `Space` — Play / Pause
- `1` — Candidate A
- `2` — Candidate B
- `3` — Original
- `4` — Removed Residual of active take

### 3. Connect an MCP Client
Configure your MCP host (Claude Desktop, Cursor, Antigravity, etc.) to launch `ambient-sound-mcp` via `uv`:

```json
{
  "mcpServers": {
    "ambient-sound": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/AmbientSound",
        "run",
        "ambient-sound-mcp"
      ],
      "env": {
        "AMBIENTSOUND_ALLOWED_ROOTS": "/absolute/path/to/your/audio/folder"
      }
    }
  }
}
```

> **Tip:** Run `uv run ambient-sound-ui` in a separate terminal while using the MCP server so any `ui_url` returned by `create_ab_test` opens directly in your browser. Both the MCP server and Web UI share `~/.ambientsound` by default (or pass `--workspace <path>` / `AMBIENTSOUND_WORKSPACE` to point them at a custom folder).

---

## Built-In MCP Tools

| Tool | Purpose |
| :--- | :--- |
| `health_check` | Verifies MCP connectivity, allowed roots, `ffmpeg`/`ffprobe`, and workspace free space. |
| `list_filters` | Returns machine-readable schemas, defaults, and tuning hints for all 14 DSP filters. |
| `create_session` | Ingests a recording into `take_000` in bounded RAM, analyzes distractions, renders overview spectrogram. |
| `list_sessions` | Lists active restoration sessions in the workspace. |
| `get_session_summary` | Returns full take tree, metrics, and A/B test history for a session. |
| `analyze_take` | Streaming analysis with configurable sensitivity, duration filters, and optional frequency band. |
| `analyze_region` | Analyzes a specific `[start_s, end_s]` window with custom sensitivity and frequency band. |
| `render_spectrogram` | Generates overview or zoomed `[start_s, end_s]` spectrogram and optional 3-panel diff PNG. |
| `process_take` | Applies an ordered filter recipe to a parent take, producing an immutable child take + residual. |
| `create_ab_test` | Registers a loudness-matched (or blind) A/B comparison and returns a ready-to-open `ui_url`. |
| `list_ab_tests` | Lists all A/B tests and verdicts in a session. |
| `get_ab_feedback` | Retrieves user verdict, artifact tags, and comments so the AI model can refine the recipe. |
| `export_take` | Exports a chosen take to WAV, FLAC, MP3, or OGG inside allowed roots. |

---

## Documentation Index

- [Agent MCP Usage Instructions](instructions.md) — standalone handover guide for AI models operating this MCP server
- [Architecture & Data Model](docs/ARCHITECTURE.md)
- [Ambient Audio Gaps & Design Decisions](docs/GAPS_AND_DESIGN.md)
- [Open-Source License Compatibility Analysis](docs/LICENSES.md)
- [Ecosystem & Model Research](docs/RESEARCH.md)
- [Roadmap](docs/ROADMAP.md)
- [Project Memory (Decisions & Invariants)](docs/MEMORY.md)
- [Engineering & DSP Tuning Notes](docs/NOTES.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
