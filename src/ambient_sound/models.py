"""Pydantic data models for audio analysis, takes, recipes, and A/B feedback."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AudioInfo(BaseModel):
    """Lightweight container metadata probed without loading sample buffers into RAM."""

    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    subtype: str
    format_name: str
    size_bytes: int


class DistractionEvent(BaseModel):
    """A localized distraction or artifact detected inside an ambient recording."""

    id: str
    start_s: float
    end_s: float
    kind: Literal[
        "mains_hum",
        "machine_drone",
        "low_freq_rumble",
        "impulsive_click",
        "transient_distraction",
        "slow_pass",
        "clipping",
    ]
    severity: Literal["low", "medium", "high"]
    freq_band_hz: tuple[float, float] | None = None
    peak_db: float | None = None
    notes: str = ""


class AudioMetrics(BaseModel):
    """Structured psychoacoustic metrics safe for JSON serialization (finite floats or None)."""

    duration_s: float
    sample_rate: int
    channels: int
    integrated_lufs: float | None = None
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    true_peak_dbtp: float | None = None
    crest_factor_db: float | None = None
    dc_offset: float | None = None
    clipped_samples: int = 0
    stereo_correlation: float | None = None
    mid_side_ratio_db: float | None = None
    spectral_centroid_hz: float | None = None
    spectral_rolloff_hz: float | None = None
    spectral_flatness: float | None = None
    rumble_ratio_db: float | None = None
    hum_peaks_hz: list[float] = Field(default_factory=list)
    tonal_peaks_hz: list[float] = Field(default_factory=list)
    loudness_range_lu: float | None = None
    noise_floor_dbfs: float | None = None
    ambient_bed_dbfs: float | None = None
    high_freq_floor_dbfs: float | None = None
    short_term_lufs_stats: dict[str, float] | None = None
    short_term_lufs_timeline: list[dict[str, float]] = Field(default_factory=list)


class StepTelemetry(BaseModel):
    """Per-step level and provenance telemetry recorded during process_take."""

    step_index: int
    filter_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_rms_dbfs: float | None = None
    output_rms_dbfs: float | None = None
    input_peak_dbfs: float | None = None
    output_peak_dbfs: float | None = None
    input_lufs: float | None = None
    output_lufs: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class FilterStep(BaseModel):
    """A single parameterized DSP filter invocation."""

    filter_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ProcessingRecipe(BaseModel):
    """Ordered DSP chain applied to a parent take."""

    parent_take_id: str
    steps: list[FilterStep]
    notes: str = ""


class TakeRecord(BaseModel):
    """Immutable audio take stored in a non-destructive session DAG."""

    take_id: str
    session_id: str
    label: str
    parent_take_id: str | None = None
    created_at: str
    audio_path: str
    residual_path: str | None = None
    spectrogram_path: str | None = None
    diff_spectrogram_path: str | None = None
    events_path: str | None = None
    sha256: str
    source_offset_s: float = 0.0
    timeline_modified: bool = False
    recipe: ProcessingRecipe | None = None
    metrics: AudioMetrics
    event_counts: dict[str, int] = Field(default_factory=dict)
    events: list[DistractionEvent] = Field(default_factory=list)
    step_metrics: list[StepTelemetry] = Field(default_factory=list)


class ABVerdict(BaseModel):
    """Human listening judgment and structured artifact feedback for an A/B test."""

    winner: Literal["a", "b", "tie", "original"]
    artifact_tags: list[str] = Field(default_factory=list)
    comment: str = ""
    submitted_at: str


class ABTestRecord(BaseModel):
    """Loudness-matched A/B listening comparison between two takes and the original."""

    ab_id: str
    session_id: str
    created_at: str
    take_a_id: str
    take_b_id: str
    original_take_id: str = "take_000"
    original_offset_s: float = 0.0
    focus_start_s: float | None = None
    focus_end_s: float | None = None
    blind: bool = False
    match_original: bool = False
    notes: str = ""
    gain_db_a: float = 0.0
    gain_db_b: float = 0.0
    gain_db_original: float = 0.0
    verdict: ABVerdict | None = None


class SessionSummary(BaseModel):
    """Full metadata state for an AmbientSound cleanup session."""

    session_id: str
    name: str
    created_at: str
    source_path: str
    takes: list[TakeRecord] = Field(default_factory=list)
    ab_tests: list[ABTestRecord] = Field(default_factory=list)
