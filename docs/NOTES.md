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
  - Use `analyze_region(..., freq_range_hz=[35, 160], sensitivity="high", min_event_s=1.5)` to locate vehicle swells cleanly without thousands of `impulsive_click` entries crowding responses. Note that natural hive crackles/ticks are often desirable ASMR detail—avoid running `declick_impulse` indiscriminately on clean hive textures.
  - For asymmetric 5–20 s vehicle passes, prefer `band_attenuate(envelope_mode="adaptive", fade_ms=1500+)` or `ambience_patch` from clean bee donors.
  - For YouTube ASMR delivery (`-14 LUFS`) from quiet (`-45` to `-55 LUFS`) sources, stage `gain` before `soft_compress` (so transients reach compressor threshold) and/or enable `loudness_normalize(target_lufs=-14.0, true_peak_ceiling_dbtp=-1.0, limit_peaks=True)` + `fade_edges`.

### 2026-11-20 — `v0.3.0` Calibration & Detector Notes
- **Bilateral Spectral Prominence:** Steep high-pass filters (`45 Hz`, `order=4`) roll off the lower sideband around `50 Hz`, causing one-sided median prominence (`peak - combined_neighbor_median`) to mistake the filter shoulder for `50 Hz` mains hum. Computing `min(peak - left_median, peak - right_median)` eliminates false hum triggers while detecting real 50/60 Hz tones.
- **Persistent Tonal Drone Separation:** Lowering tonal prominence to `> 7.0 dB` catches steady `7–11 dB` machine drones, while requiring bilateral prominence `>= 4.5 dB` in `>= 60%` of time segments separates continuous drones across long files from brief 15 s passing vehicles.
- **Detector Floor `-92.0 dBFS`:** Quiet `-50 LUFS` field recordings filtered into narrow sub-bands routinely fall below `-55` to `-60 dBFS`. Moving the noise gate floor to `-92.0 dBFS` restores sensitivity and band-limited event discovery on quiet sources.
