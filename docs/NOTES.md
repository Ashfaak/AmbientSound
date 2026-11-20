# Engineering & DSP Tuning Notes (`NOTES.md`)

Working notes on DSP design choices, tuning heuristics for ambient soundscapes, and session experiments.

---

## 1. DSP Implementation Notes (v0.1.0)

### Linked-Stereo Spectral Processing
- Computing independent spectral masks per channel causes subtle stereo shimmer ("phasiness") in wide stereo field recordings (rain, rustling leaves, ocean surf).
- Both `spectral_denoise` and `band_attenuate` compute a single cross-channel magnitude estimate (`mean` across channels), derive one smooth 2D `(freq_bins, frames)` gain mask, and multiply all channels by the exact same gain matrix.
- This keeps inter-channel level/phase relationships undisturbed.

### Zero-Phase Filtering (`sosfiltfilt` / `filtfilt`)
- `highpass_rumble` and `dehum_notch` use forward-backward filtering (`sosfiltfilt` / `filtfilt`) so low-frequency phase response stays linear and exact sample alignment with the parent take is preserved for `residual = parent - processed`.
- Note: Filter order in `sosfiltfilt` doubles effective rolloff slope (`order=4` -> 48 dB/octave equivalent). Keep `order=2` or `order=4` depending on how steep the wind cutoff needs to be.

### `ambience_patch` Auto-Donor Selection
- When `source_start_s` is omitted, `ambience_patch` scans sliding windows across the recording outside `[start_s, end_s]` (with a padding margin) and scores candidate windows by:
  - Peak-to-RMS ratio (penalizing spiky transients)
  - High-frequency derivative peaks (penalizing clicks)
  - Closeness of RMS to the median ambient bed
- Boundary matching scales donor amplitude subtly toward local boundary RMS and applies equal-power (`cos`/`sin`) crossfade edges (`crossfade_ms`, default `40 ms`) to prevent clicks or abrupt texture steps.

---

## 2. Ambient Tuning Heuristics for the AI Agent

| Recording Environment | Common Distractions | Recommended First Pass | Watch Out For (Residual Check) |
| :--- | :--- | :--- | :--- |
| **Rain / Surf** | Subsonic wind buffeting, stray traffic horn, mic drips | `highpass_rumble(55 Hz)` + `declick_impulse` + `ambience_patch` on horn | Heavy `spectral_denoise` turns rain into metallic static — avoid unless constant hiss exists. |
| **Forest / Birds** | Distant road hum, footsteps, cough, wind gust | `highpass_rumble(70 Hz)` + `band_attenuate` (`<250 Hz` during car pass) or `ambience_patch` | Keep `band_attenuate` below bird frequencies (`>1.5 kHz`) untouched. |
| **Indoor Room Tone / Café** | 50/60 Hz HVAC/mains hum, chair scrape, clear throat | `dehum_notch(50 or 60 Hz, harmonics=4)` + `ambience_patch` on scrape | Compare 50 Hz vs 60 Hz prominence in `analyze_take` (`hum_peaks_hz`) before notching! |
| **Windy Exterior** | Heavy `<120 Hz` gusts, mic diaphragm pop | `highpass_rumble(85-110 Hz)` + `declick_impulse` | If gusts saturate the mic (`clipping`), `ambience_patch` works better than filtering alone. |

---

## 3. Experiment Log

### 2026-11-20 — Synthetic Demo & v0.2.0 Streaming Architecture Notes
- Demo fixture simulates: stereo pink-noise rain bed + `60 Hz` + `120 Hz` hum + impulsive mic click + `950 Hz` distraction burst (`2.4s–3.1s`).
- Candidate A compares surgical `band_attenuate` (`750–1200 Hz`, `-22 dB`) against Candidate B `ambience_patch` (`2.35s–3.15s`).
- Listen to `Residual` (`hotkey 4`) in the A/B Studio to verify rain texture outside the distraction stays intact.
- **Bee-Hive & Passing-Vehicle Notes (`ZOOM0001.WAV` class):**
  - Honeybee hive buzzing concentrates fundamental energy around `200–300 Hz` plus upper harmonics, overlapping engine/tyre passes (`80–1000 Hz`).
  - Use `analyze_region(..., freq_range_hz=[35, 160], sensitivity="high")` to locate vehicle swells below the hive fundamental without false triggers, then preview `trim_range` on a `30 s` slice before executing full-file batched `band_attenuate(envelope_mode="swell", fade_ms=1500+)` or `cut_regions(crossfade_ms=1000+)`.
  - For YouTube ASMR delivery (`-14 LUFS`), chain gentle `parametric_eq` (`lowshelf` warmth + `>10 kHz` `highshelf` air), `soft_compress` (`ratio 1.8`), `loudness_normalize(target_lufs=-14.0)`, `true_peak_limiter(ceiling_dbtp=-1.0)`, and `fade_edges`.
