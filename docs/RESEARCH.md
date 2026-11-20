# Ecosystem & Open-Source Model Research

This document captures technical findings from reviewing existing MCP audio projects and open-source ML/DSP models relevant to **AmbientSound**.

---

## 1. Open-Source MCP Audio Ecosystems Reviewed

### `djynnius/ferraus` (Rust, `Apache-2.0 OR MIT`)
- **Design strengths:** Single CLI + native MCP server where every tool call emits structured JSON and writes deterministic provenance sidecars (`<output>.ferraus.json`).
- **Relevance to AmbientSound:** Strongest architectural & license match. Worth integrating later as an optional backend for heavy ONNX stem/repair workloads (`ferraus-restore`, `ferraus-stems`) if Rust/ONNX acceleration is needed.

### `GeekatplayStudio/music-suite` (Python + Next.js, `MIT`)
- **Design strengths:** Local-first loopback UI paired with an MCP server that defaults to read-only operations and gates mutations behind an explicit environment variable.
- **Relevance to AmbientSound:** Directly inspired our local-only `127.0.0.1` A/B studio pairing and safe sandbox defaults (note: GPL-3.0 audio libraries used in parts of that ecosystem are strictly excluded here).

### `psyb0t/docker-audiolla` (Docker/Python, `WTFPL`)
- **Design strengths:** Multi-engine restoration/DSP toolbox exposing HTTP + MCP endpoints, plus a thoughtful runtime safety pattern that refuses to load non-commercial weights unless the operator explicitly opts in via environment flag.
- **Relevance to AmbientSound:** We adopted the explicit opt-in guardrail (`AMBIENTSOUND_ALLOW_NONCOMMERCIAL=0`) for future ML weight downloads while staying strictly inside permissive Apache-2.0/MIT/BSD libraries.

### `fadelabs/phantom` (Python, `AGPL-3.0`)
- **Design strengths:** Demonstrates how giving an LLM structured acoustic measurements (LUFS, tonal balance, stereo correlation, phase, frequency masking) enables iterative audio engineering over MCP rather than blind parameter guessing.
- **Relevance to AmbientSound:** Conceptual inspiration only (excluded from any code borrowing due to AGPL-3.0). We implemented original ambient-focused metrics (rumble ratio, 50/60 Hz harmonic hum prominence, impulsive click detection, transient event spans, and removed-residual diffing).

---

## 2. Open-Source ML Models for Ambient Distraction Cleanup (Planned Phases 2–3)

Because **AmbientSound** runs initially on CPU-first hardware (no dedicated CUDA GPU), v0.1 focuses on fast deterministic DSP + psychoacoustic metrics. Future optional phases can layer in open-source ML models:

| Model / Library | Capability | License | Hardware Fit | Ambient Role |
| :--- | :--- | :--- | :--- | :--- |
| **Silero VAD** | Lightweight speech activity timestamps | MIT | CPU (<1 ms/frame ONNX) | Flags human speech bursts in nature/room recordings so they can be attenuated or patched out. |
| **YAMNet** | 521-class AudioSet sound event classification | Apache-2.0 | CPU (TFLite / ONNX) | Automatically tags distraction spans (`Speech`, `Siren`, `Bark`, `Car horn`, `Cough`) vs ambience (`Rain`, `Wind`, `Stream`). |
| **PANNs (CNN14 / Tagging)** | High-accuracy audio tagging & frame-level SED | MIT | CPU / GPU | Richer distraction labeling for field recordings. |
| **DeepFilterNet** | Neural speech enhancement | MIT / Apache-2.0 | CPU (Rust/ONNX) | **Inverted usage:** extract the speech stem $S$ and subtract $\alpha S$ from the mix (with ambience repair on severe gaps), turning a speech enhancer into a speech *remover*. |
| **AudioSep** | Text-queried open-domain sound separation | MIT | GPU recommended (slow CPU) | Allows prompts like `"remove siren"` or `"isolate distant chatter"` to build a targeted distraction mask. |
| **CLAPSep** | Multimodal query-conditioned separation | MIT | GPU recommended | Alternative zero-shot target sound extractor using LAION-CLAP embeddings. |
| **SAM Audio** | Text/span/video target sound segmentation | SAM License (restricted field-of-use) | GPU | Optional future plugin only behind explicit `AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1` (or restricted-license opt-in) acknowledgment; never bundled in core. |
