# Ambient Audio Gaps & Design Decisions

Generic audio cleanup software is heavily biased toward **speech and music**: it treats rain, wind, room tone, and outdoor texture as unwanted noise and tries to isolate dry voice. **AmbientSound** flips that assumption and solves eight concrete engineering gaps:

---

## 1. The Ambient Inverse Problem
- **Gap:** Standard denoisers treat continuous broadband background sound as noise, leaving hollow, water-damaged, or overly gated ambience.
- **Design:** AmbientSound treats the continuous acoustic bed as the **signal** and targets **localized distractions**: mains hum (`50/60 Hz` harmonics), subsonic wind rumble (`<80 Hz`), mic handling clicks, and sudden transients (sirens, horns, barks, stray speech).

## 2. Patch, Don't Mute ("Room Tone Synthesis")
- **Gap:** Zeroing out or hard-gating a cough or car horn creates an unmistakable "vacuum dropout" in rain, forest, or café recordings.
- **Design:** `ambience_patch` locates the cleanest low-anomaly segment elsewhere in the recording (or uses a user-specified `source_start_s`), tiles it smoothly if longer duration is needed, matches boundary RMS, and crossfades it into the distracted span with equal-power (`cos`/`sin`) edges.

## 3. Giving the LLM Ears & Eyes
- **Gap:** Text-only LLMs cannot directly hear audio files and guess blind DSP parameters when given only file paths.
- **Design:**
  - `analyze_take` emits structured acoustic metrics + timestamped `DistractionEvent` spans (`mains_hum`, `low_freq_rumble`, `impulsive_click`, `transient_distraction`, `clipping`) with dominant `freq_band_hz` and severity.
  - `render_spectrogram` creates dual waveform+spectrogram PNGs and 3-panel Before/After/Removed-Residual PNGs for visual inspection.
  - `residual.wav` isolates **only what was removed**, letting both model and listener detect if real ambience was accidentally stripped.

## 4. Bias-Free Human A/B Listening & Feedback Loop
- **Gap:** Playing two exported WAV files sequentially in a media player introduces loudness bias, seek-time memory decay, and unstructured feedback ("sounds weird").
- **Design:** The FastAPI + Web Audio API studio offers:
  - Instant sample-locked `12 ms` crossfade switching (`1`, `2`, `3`, `4` hotkeys).
  - Attenuation-only LUFS loudness matching.
  - Loop focus spans around edited regions.
  - Blind Mode (`blind=true`) hiding labels and recipes until verdict submission.
  - Predefined artifact chips (`Metallic / swishy`, `Hollow gap`, `Stereo image shifted`, etc.) + freeform comments captured back into the session via `get_ab_feedback`.

## 5. Stereo & Spatial Coherence
- **Gap:** Running independent mono spectral gates on L/R channels causes stereo image flutter, phase warble, and phantom-center collapse on stereo field recordings.
- **Design:** Both `spectral_denoise` and `band_attenuate` compute **linked-stereo gain masks** across channels (`mean` magnitude envelope -> unified smooth gain applied equally to L and R), preserving binaural/ORTF/XY width, tracked via `stereo_correlation` and `mid_side_ratio_db`.

## 6. Hardware-Realistic Tiering
- **Gap:** Assuming CUDA GPUs breaks local usability on CPU-only laptops (such as Intel Iris Xe systems).
- **Design:** Tier 1 (v0.1) runs 100% deterministic CPU DSP (`numpy`/`scipy`) in milliseconds to seconds; Tier 2 adds lightweight CPU ONNX classifiers; Tier 3 makes heavy GPU separation optional.

## 7. Long-Form Field Recordings
- **Gap:** Field recordings are often 30–120 minutes long; loading huge uncompressed arrays or processing full files while iterating wastes time.
- **Design:** v0.1 encourages auditioning focused short clips / regions during rapid A/B tuning (`focus_start_s..focus_end_s`), with overlap-add chunked streaming designed for Phase 4.

## 8. MCP Path Sandboxing & License Safety
- **Gap:** Unbounded MCP file paths allow reading/writing arbitrary system paths, and bundled ML weights often hide incompatible non-commercial licenses.
- **Design:** `ensure_path_allowed` confines all file reads/exports to `AMBIENTSOUND_ALLOWED_ROOTS`, stdout stays strictly reserved for MCP JSON-RPC, and non-commercial ML plugins require `AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1`.
