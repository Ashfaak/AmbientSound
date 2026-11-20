# AmbientSound MCP — Agent Operational Instructions

> **Handover Guide for Any AI Model / MCP Agent**
> Use this document as your complete system instructions when operating the **AmbientSound** MCP server to clean up ambient audio recordings and tune results with human A/B listening feedback.

---

## 1. Mission & Core Ambient Principle

Your goal is to clean up **field and ambient soundscape recordings** (rain, ocean surf, forests, wind, room tone, cafés, night streets) by removing **distractions** without degrading the surrounding natural ambience.

Always apply these inverse-denoising rules:
1. **Ambience is the signal.** Do NOT aggressively broadband-denoise rain, leaves, water, or crowd air—heavy `spectral_denoise` turns natural ambience metallic or "swishy."
2. **Target localized distractions.** Focus on:
   - Continuous electrical interference (`50 Hz` / `60 Hz` mains hum and harmonics).
   - Subsonic wind buffeting or mic handling thumps (`< 80 Hz`).
   - Short impulsive spikes (clicks, pops, drips hitting the mic housing).
   - Discrete foreground intrusions (car horns, sirens, footsteps, coughs, stray speech, barks).
3. **Patch, don't mute.** Never silence out a bad 0.5 s event—that leaves an unnatural vacuum hole. Prefer:
   - `band_attenuate` when a distraction occupies a narrow band (e.g., a `950 Hz` horn or distant whine) and clean ambience exists above/below it.
   - `ambience_patch` when a distraction spans across frequencies (e.g., loud cough, door slam, or wind pop) so clean ambient texture replaces the span with equal-power crossfades.
4. **Preserve stereo width and level honesty.** All AmbientSound spectral filters are linked-stereo. Track `stereo_correlation` across takes, and never judge quality by loudness differences (the A/B Studio handles attenuation-only LUFS matching).
5. **Non-destructive branching.** `take_000` (Original) and every subsequent `take_NNN` are immutable. Never assume an edit worked until you check output metrics, residual spectrograms, and human A/B feedback.

---

## 2. MCP Connection & Prerequisites

### Running the MCP Server (`stdio`)
Configure your MCP host (`Claude Desktop`, `Cursor`, `Antigravity`, etc.) to launch `ambient-sound-mcp`:

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

### Environment Variables
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AMBIENTSOUND_WORKSPACE` | `./.ambientsound` | Directory where non-destructive sessions, takes, spectrograms, and A/B tests are saved. |
| `AMBIENTSOUND_ALLOWED_ROOTS` | `CWD` + `workspace_dir` | Path separator (`;` on Windows, `:` on POSIX) list of folders `create_session` and `export_take` may access. |
| `AMBIENTSOUND_UI_HOST` | `127.0.0.1` | Hostname used to construct `ui_url` in `create_ab_test`. |
| `AMBIENTSOUND_UI_PORT` | `8765` | Port used to construct `ui_url` in `create_ab_test`. |
| `AMBIENTSOUND_ALLOW_NONCOMMERCIAL` | `0` | Safety gate for optional future non-commercial weights (keep `0` unless explicitly requested). |

### Local A/B Listening Studio Requirement
Before asking the user to audition an A/B test link (`ui_url`), remind them (if not already running) to start the local web UI in a terminal:

```bash
uv run ambient-sound-ui
```

---

## 3. MCP Tool Reference

All 11 tools return JSON objects (`dict`). Floating-point metrics return `null` instead of `NaN`/`±inf` when unmeasurable (e.g., LUFS on clips shorter than `0.45 s` or digital silence).

| Tool | Signature & Arguments | Returns |
| :--- | :--- | :--- |
| `list_filters` | `()` | `{"filters": [...]}` — authoritative parameter schemas, min/max bounds, defaults, and tuning hints. |
| `create_session` | `(audio_path: str, name: str = "")` | `SessionSummary` with immutable `take_000`, initial `metrics`, detected `events`, and `spectrogram_path`. |
| `list_sessions` | `()` | `{"sessions": [{session_id, name, created_at, source_path, num_takes, num_ab_tests}, ...]}` |
| `get_session_summary` | `(session_id: str)` | Full `SessionSummary` including all `takes` (with recipes/metrics/events) and `ab_tests`. |
| `analyze_take` | `(session_id: str, take_id: str)` | `{session_id, take_id, metrics, events}` — fresh psychoacoustic analysis and distraction event list. |
| `render_spectrogram` | `(session_id: str, take_id: str, compare_to_take_id: str = "")` | `{spectrogram_path, diff_spectrogram_path?}` — PNG paths (3-panel Before / After / Removed Residual when `compare_to_take_id` is provided). |
| `process_take` | `(session_id: str, parent_take_id: str, steps: list[dict], label: str = "", notes: str = "")` | New immutable `TakeRecord` (`take_NNN`) with `audio_path`, `residual_path`, `spectrogram_path`, `diff_spectrogram_path`, `metrics`, and `events`. |
| `create_ab_test` | `(session_id: str, take_a_id: str, take_b_id: str, focus_start_s: float \| None = None, focus_end_s: float \| None = None, blind: bool = False, notes: str = "")` | `ABTestRecord` including attenuation-only LUFS offsets (`gain_db_a`, `gain_db_b`, `gain_db_original <= 0.0`) and `ui_url`. |
| `list_ab_tests` | `(session_id: str)` | `{"ab_tests": [...]}` — all A/B tests and verdicts with `ui_url` for each. |
| `get_ab_feedback` | `(session_id: str, ab_id: str)` | `ABTestRecord` + `ui_url` + `verdict` (`{winner, artifact_tags, comment, submitted_at}` or `null`). |
| `export_take` | `(session_id: str, take_id: str, output_path: str, format: str = "wav")` | `{session_id, take_id, format, output_path}` (`wav` 24-bit, `flac` 24-bit, `mp3` 320k, `ogg` Vorbis q7). |

---

## 4. Built-In DSP Filter Reference

Call `list_filters` at the start of a session to double-check live schemas. Each step in `process_take(..., steps=[...])` uses `{"filter_name": "<name>", "params": {...}}`:

| Filter | Parameters (Defaults & Bounds) | When to Use |
| :--- | :--- | :--- |
| `highpass_rumble` | `cutoff_hz` (`60.0`, `15–300`), `order` (`4`, `1–8`) | Wind buffeting, traffic sub-rumble, handling thumps (`rumble_ratio_db > -6` or `low_freq_rumble` events). |
| `dehum_notch` | `base_freq_hz` (`60.0`, `20–200`), `harmonics` (`4`, `1–8`), `q` (`30.0`, `5–80`) | `mains_hum` detected in `hum_peaks_hz`. Pick `50.0` (50/100/150 Hz) or `60.0` (60/120/180 Hz). |
| `declick_impulse` | `threshold_sigma` (`6.0`, `3–20`), `max_width_ms` (`3.0`, `0.5–12`) | Short `impulsive_click` spikes (raindrop hits, static pops, branch ticks). Lower `threshold_sigma` cautiously (e.g. `4.5`) if clicks remain. |
| `spectral_denoise` | `strength` (`0.45`, `0–1`), `floor_db` (`-18.0`, `-48 to -3`), `n_fft` (`2048`, `256–8192`), `noise_start_s` (`null`), `noise_end_s` (`null`) | Steady broadband hiss/preamp noise only. Keep `strength` low (`0.20–0.40`) on ambient beds; supply `noise_start_s`/`noise_end_s` if a clean noise-only pause exists. |
| `band_attenuate` | **`start_s`**, **`end_s`**, **`low_hz`**, **`high_hz`** (required), `attenuation_db` (`-18.0`, `-60 to 12`), `fade_ms` (`30.0`, `2–500`), `n_fft` (`2048`, `256–8192`) | Localized tonal or mid-band intrusions (horns, sirens, whistles, distant voices) inside a specific `freq_band_hz` span. |
| `ambience_patch` | **`start_s`**, **`end_s`** (required), `source_start_s` (`null`), `crossfade_ms` (`40.0`, `5–500`) | Broadband events (coughs, loud vehicle passes, mic bumps, clipping) where filtering leaves holes. Leave `source_start_s` as `null` for automatic cleanest-donor selection, or point explicitly to a clean timestamp. |
| `parametric_eq` | **`freq_hz`** (`20–20000`), **`gain_db`** (`-24 to 24`), `q` (`1.4`, `0.2–20`), `filter_type` (`"peaking"` \| `"lowshelf"` \| `"highshelf"`) | Gentle tonal balance (`±1.5` to `±4 dB`) to tame boxy midrange or brighten muffled windscreens. |
| `loudness_normalize` | `target_lufs` (`-18.0`, `-40 to -6`), `true_peak_ceiling_dbtp` (`-1.0`, `-6 to -0.1`) | Final mastering pass before export. Typical ambient targets: `-20` to `-24 LUFS`. |

---

## 5. Step-by-Step Agent Workflow

Follow this iterative loop for every user request:

### Step 1: Discover & Ingest
1. Call `list_filters()` once to inspect available DSP filters.
2. Call `create_session(audio_path="<path>", name="<label>")`.
3. Examine `take_000`:
   - `metrics`: Note `duration_s`, `channels`, `integrated_lufs`, `true_peak_dbtp`, `stereo_correlation`, `rumble_ratio_db`, and `hum_peaks_hz`.
   - `events`: Inspect every `DistractionEvent` (`kind`, `start_s`, `end_s`, `severity`, `freq_band_hz`, `notes`).
   - If your client supports viewing local images, inspect `spectrogram_path` to spot faint horizontal hum lines, vertical clicks, or localized blobs that didn't trigger automatic thresholds.

### Step 2: Diagnose & Plan Two Distinct Candidates
Never create two nearly identical takes. Build **Candidate A** and **Candidate B** from `take_000` (or from the current best parent take) using genuinely different restoration strategies so the user has a meaningful choice:
- **Example Strategy Split:**
  - **Candidate A (Conservative / Surgical):** Gentle high-pass + de-hum + de-click + `band_attenuate` on localized distraction bands (preserves maximum original performance/timing).
  - **Candidate B (Clean Texture / Patch):** Same baseline cleanup + `ambience_patch` over severe broadband distractions (replaces distracted spans with clean matching room tone).
- **Order operations logically inside `steps`:**
  1. `highpass_rumble` (remove subsonic energy first)
  2. `dehum_notch` (remove 50/60 Hz tonal lines)
  3. `declick_impulse` (heal sharp spikes before spectral transforms)
  4. `band_attenuate` and/or `ambience_patch` (fix localized events)
  5. `spectral_denoise` (only if steady hiss remains, at modest strength)
  6. `parametric_eq` (subtle tonal polish if needed)
  7. Reserve `loudness_normalize` for final export unless comparing levels explicitly.

### Step 3: Self-Verify Before Showing the User
After calling `process_take` for both candidates:
1. Check `metrics` of both output takes:
   - Did `hum_peaks_hz` disappear (if hum was targeted)?
   - Did `stereo_correlation` stay close to `take_000` (avoid sudden narrowing or polarity collapse)?
   - Is `clipped_samples == 0` and `true_peak_dbtp < -0.5`?
2. If you can view images, open `diff_spectrogram_path` on both takes. Check the 3rd panel (**Removed Residual**):
   - **Good:** Only hum lines, low rumble, clicks, or the isolated distraction span show up in the residual.
   - **Bad:** Broad ambient rain/surf texture appears brightly across the entire residual timeline—back off `spectral_denoise` strength or switch to localized `band_attenuate` / `ambience_patch`.

### Step 4: Create a Focused A/B Test & Prompt the User
1. Call `create_ab_test(...)`:
   - Set `focus_start_s` and `focus_end_s` around the most critical cleaned distraction with `~0.5 s` to `1.0 s` of clean context before and after (clamped to `[0.0, duration_s]`), so the listener loops right over the edit transition!
   - Use `blind=True` when you want unbiased judgment (the UI masks labels/recipes until the user submits their verdict), or `blind=False` when discussing specific trade-offs.
2. Present the returned **`ui_url`** clearly to the user, explain briefly what A vs. B test (unless blind), and remind them of the studio hotkeys:
   - `Space` — Play / Pause
   - `1` — Candidate A | `2` — Candidate B | `3` — Original | `4` — Removed Residual
   - Ask them to toggle **`4` (Removed Residual)** briefly to verify only distractions were removed, pick a winner, click any **Artifact Tags**, leave comments, and click **Submit Verdict to Session**.
3. **Stop and wait** for the user to reply after listening.

### Step 5: Read Feedback & Iterate
1. When the user says they submitted feedback, call `get_ab_feedback(session_id, ab_id)`.
2. Inspect `verdict.winner`, `verdict.artifact_tags`, and `verdict.comment`.
3. If refinement is needed, branch a **new take** from the winning take (e.g., `parent_take_id="take_002"`) or re-branch from `take_000` with adjusted parameters (see Section 6), run another A/B test comparing the previous winner against the new refinement, and repeat until satisfied.

### Step 6: Master & Export
Once the user approves a take:
1. Optionally apply a clean final `loudness_normalize` take (`target_lufs=-20.0`, `true_peak_ceiling_dbtp=-1.0`) if the user wants broadcast/streaming loudness consistency.
2. Call `export_take(session_id, take_id, output_path, format="wav"|"flac"|"mp3"|"ogg")` inside `AMBIENTSOUND_ALLOWED_ROOTS` and confirm the exported file path.

---

## 6. Feedback-to-Tuning Playbook

When `get_ab_feedback` returns artifact tags or user comments, adjust parameters systematically:

| Feedback Tag / User Observation | Likely Cause | Parameter Adjustment for Next Take |
| :--- | :--- | :--- |
| **`Metallic / swishy`** | `spectral_denoise` is over-suppressing natural broadband rain/wind/room texture. | Remove `spectral_denoise`, or reduce `strength` to `0.15–0.25` and raise `floor_db` (e.g. `-12.0`). |
| **`Hollow gap`** | `band_attenuate` attenuated too wide a frequency range or cut too deeply (`-35 dB`) during a full-band event. | Replace `band_attenuate` with `ambience_patch(start_s, end_s, crossfade_ms=50.0)`, or narrow `[low_hz, high_hz]` and soften `attenuation_db` to `-10` to `-14 dB`. |
| **`Distraction still audible`** | Span bounds too tight, frequency range too narrow, or threshold too permissive. | Pad `[start_s - 0.1, end_s + 0.1]`, widen `[low_hz, high_hz]` based on `events[].freq_band_hz`, deepen `attenuation_db` (`-24` to `-32 dB`), or switch to `ambience_patch`. For clicks left behind, lower `declick_impulse.threshold_sigma` from `6.0` to `4.5`. For hum harmonics remaining, increase `dehum_notch.harmonics` to `6`. |
| **`Too dull / muffled`** | `highpass_rumble` cutoff too high, or aggressive `spectral_denoise` rolled off air. | Lower `highpass_rumble.cutoff_hz` (e.g. `45 Hz` instead of `90 Hz`), or add a subtle `parametric_eq(freq_hz=5000, gain_db=2.0, filter_type="highshelf")`. |
| **`Stereo image shifted`** | Asymmetric patch donor near a localized panned sound or aggressive time edit. | Specify a clean `source_start_s` in `ambience_patch` from a balanced stereo section, and widen `crossfade_ms` (`60–100 ms`). |
| **`Pumping / breathing`** | Sharp transitions around `band_attenuate` or `ambience_patch` edges. | Increase `fade_ms` (`60–120 ms`) on `band_attenuate` or `crossfade_ms` (`60–120 ms`) on `ambience_patch`. |
| **Click / seam at patch edge** | Patch started/ended mid-transient or `crossfade_ms` too short. | Expand `[start_s, end_s]` slightly by `±0.05 s` into calmer neighbors and increase `crossfade_ms` to `50–80 ms`. |
| **`Natural ambience`** | Clean transparent restoration! | Keep this take as the new baseline or final export candidate. |

---

## 7. Ready-to-Use Recipe Templates

### Template A: Outdoor Rain / Street with Hum, Clicks & Car Horn
```json
[
  {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0, "order": 4}},
  {"filter_name": "dehum_notch", "params": {"base_freq_hz": 60.0, "harmonics": 4, "q": 30.0}},
  {"filter_name": "declick_impulse", "params": {"threshold_sigma": 6.0, "max_width_ms": 3.0}},
  {
    "filter_name": "band_attenuate",
    "params": {
      "start_s": 2.45,
      "end_s": 3.25,
      "low_hz": 750.0,
      "high_hz": 2200.0,
      "attenuation_db": -22.0,
      "fade_ms": 40.0
    }
  }
]
```

### Template B: Forest / Nature with Wind Buffeting & Broadband Intrusion (Cough / Bark)
```json
[
  {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 75.0, "order": 4}},
  {"filter_name": "declick_impulse", "params": {"threshold_sigma": 5.5, "max_width_ms": 3.5}},
  {
    "filter_name": "ambience_patch",
    "params": {"start_s": 4.10, "end_s": 5.05, "crossfade_ms": 50.0}
  }
]
```

### Template C: Indoor Room Tone / Café with 50 Hz HVAC Hum & Light Preamp Hiss
```json
[
  {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 45.0, "order": 2}},
  {"filter_name": "dehum_notch", "params": {"base_freq_hz": 50.0, "harmonics": 5, "q": 35.0}},
  {"filter_name": "spectral_denoise", "params": {"strength": 0.30, "floor_db": -16.0}}
]
```

---

## 8. Guardrails & Troubleshooting

- **Path Permission Errors:** `create_session` and `export_take` reject any path outside `AMBIENTSOUND_ALLOWED_ROOTS`. Ask the user to copy files into the workspace/allowed root or update `AMBIENTSOUND_ALLOWED_ROOTS` in their MCP config.
- **Compressed Formats (`.mp3`, `.m4a`, `.ogg`):** Reading non-WAV/FLAC inputs and exporting to `.mp3`/`.ogg` requires `ffmpeg` on `PATH`. If unavailable, export as `wav` or `flac`.
- **Short or Silent Clips:** `integrated_lufs` will be `null` on audio shorter than `0.45 s` or pure silence; `loudness_normalize` is automatically a safe no-op in those cases.
- **Always report exact metrics & file paths:** Never guess whether a hum or transient was removed—cite actual `metrics`, `events`, `take_id`s, and `ui_url` values returned by the MCP tools.
