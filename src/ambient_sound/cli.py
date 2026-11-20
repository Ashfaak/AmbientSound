"""Command-line interface for AmbientSound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ambient_sound.analysis import analyze_audio
from ambient_sound.config import Settings
from ambient_sound.io_audio import load_audio, save_wav
from ambient_sound.mcp_server import main as mcp_main
from ambient_sound.session import SessionManager
from ambient_sound.web_ui import run_ui


def _generate_synthetic_ambient_demo(out_path: Path, sample_rate: int = 44100) -> Path:
    """Generate a realistic synthetic stereo ambient bed with hum, click, and distraction."""
    rng = np.random.default_rng(42)
    duration_s = 6.0
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    # Stereo uncorrelated pink-ish rain bed via 1-pole lowpass shaping on white noise
    white = rng.normal(0.0, 0.05, size=(n_samples, 2))
    rain = np.empty_like(white)
    for ch in range(2):
        acc = 0.0
        for i in range(n_samples):
            acc = 0.94 * acc + 0.06 * white[i, ch]
            rain[i, ch] = acc * 2.2

    # Subtle shared stereo ambience center
    shared = 0.25 * (rain[:, 0] + rain[:, 1])[:, np.newaxis]
    bed = 0.85 * rain + shared

    # 60 Hz + 120 Hz mains hum
    hum = (
        0.018 * np.sin(2.0 * np.pi * 60.0 * t) + 0.008 * np.sin(2.0 * np.pi * 120.0 * t)
    )[:, np.newaxis]

    # Impulsive mic click around 1.20 s
    click = np.zeros((n_samples, 2), dtype=np.float64)
    c_idx = int(1.20 * sample_rate)
    click[c_idx : c_idx + 14, 0] = 0.65 * np.exp(-np.arange(14) / 2.5)
    click[c_idx : c_idx + 14, 1] = 0.62 * np.exp(-np.arange(14) / 2.5)

    # Localized mid-band horn/tone distraction between 2.50 s and 3.20 s
    burst = np.zeros((n_samples, 2), dtype=np.float64)
    b0 = int(2.50 * sample_rate)
    b1 = int(3.20 * sample_rate)
    env = np.hanning(b1 - b0)
    tone = 0.12 * env * (
        np.sin(2.0 * np.pi * 950.0 * t[b0:b1]) + 0.5 * np.sin(2.0 * np.pi * 1900.0 * t[b0:b1])
    )
    burst[b0:b1, 0] = tone
    burst[b0:b1, 1] = 0.92 * tone

    mix = np.clip(bed + hum + click + burst, -0.98, 0.98)
    return save_wav(out_path, mix, sample_rate, subtype="FLOAT")


def run_demo(settings: Settings) -> dict[str, str]:
    """Create a synthetic demo session, two cleanup takes, and an A/B listening test."""
    demo_wav = settings.workspace_dir / "demo_rain_street.wav"
    _generate_synthetic_ambient_demo(demo_wav)

    manager = SessionManager(settings)
    summary = manager.create_session(demo_wav, name="Demo Rain Street")

    take_a = manager.process_take(
        session_id=summary.session_id,
        parent_take_id="take_000",
        label="Surgical Band Attenuate",
        notes="Highpass 55 Hz + 60 Hz de-hum + declick + localized 750-2200 Hz attenuation.",
        steps=[
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0, "order": 4}},
            {
                "filter_name": "dehum_notch",
                "params": {"base_freq_hz": 60.0, "harmonics": 4, "q": 30.0},
            },
            {
                "filter_name": "declick_impulse",
                "params": {"threshold_sigma": 6.0, "max_width_ms": 3.0},
            },
            {
                "filter_name": "band_attenuate",
                "params": {
                    "start_s": 2.45,
                    "end_s": 3.25,
                    "low_hz": 750.0,
                    "high_hz": 2200.0,
                    "attenuation_db": -26.0,
                    "fade_ms": 35.0,
                },
            },
        ],
    )

    take_b = manager.process_take(
        session_id=summary.session_id,
        parent_take_id="take_000",
        label="Ambience Texture Patch",
        notes="Highpass 55 Hz + de-hum + declick + clean rain bed patch over 2.45-3.25s.",
        steps=[
            {"filter_name": "highpass_rumble", "params": {"cutoff_hz": 55.0, "order": 4}},
            {
                "filter_name": "dehum_notch",
                "params": {"base_freq_hz": 60.0, "harmonics": 4, "q": 30.0},
            },
            {
                "filter_name": "declick_impulse",
                "params": {"threshold_sigma": 6.0, "max_width_ms": 3.0},
            },
            {
                "filter_name": "ambience_patch",
                "params": {"start_s": 2.45, "end_s": 3.25, "crossfade_ms": 45.0},
            },
        ],
    )

    ab = manager.create_ab_test(
        session_id=summary.session_id,
        take_a_id=take_a.take_id,
        take_b_id=take_b.take_id,
        focus_start_s=1.8,
        focus_end_s=3.8,
        blind=False,
        notes="Compare spectral band attenuation (A) vs clean ambient texture patch (B).",
    )
    return {
        "session_id": summary.session_id,
        "ab_id": ab.ab_id,
        "ui_url": manager.ab_ui_url(summary.session_id, ab.ab_id),
        "demo_wav": str(demo_wav),
    }


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ambient-sound",
        description="AmbientSound: MCP ambient audio restoration & A/B listening studio",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("mcp", help="Run the stdio MCP server")

    ui_parser = sub.add_parser("ui", help="Run the local A/B listening studio web server")
    ui_parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    ui_parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8765)")
    ui_parser.add_argument(
        "--workspace",
        default=None,
        help="Session workspace directory (default: ~/.ambientsound or AMBIENTSOUND_WORKSPACE)",
    )

    analyze_parser = sub.add_parser("analyze", help="Analyze an audio file and print JSON metrics")
    analyze_parser.add_argument("audio_path", help="Path to audio file")

    sub.add_parser("demo", help="Generate a synthetic demo session and A/B test")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "mcp":
        mcp_main()
        return

    if args.command == "ui":
        if args.workspace:
            ws = Path(args.workspace)
            settings = Settings(
                workspace_dir=ws,
                allowed_roots=[*settings.allowed_roots, ws],
                allow_noncommercial=settings.allow_noncommercial,
                ui_host=settings.ui_host,
                ui_port=settings.ui_port,
            )
        run_ui(
            host=args.host or settings.ui_host,
            port=args.port or settings.ui_port,
            settings=settings,
        )
        return

    if args.command == "analyze":
        audio, sr = load_audio(Path(args.audio_path).expanduser().resolve())
        metrics, events = analyze_audio(audio, sr)
        print(
            json.dumps(
                {
                    "metrics": metrics.model_dump(),
                    "events": [e.model_dump() for e in events],
                },
                indent=2,
            )
        )
        return

    if args.command == "demo":
        result = run_demo(settings)
        print("Created AmbientSound synthetic demo session:")
        print(f"  Session ID : {result['session_id']}")
        print(f"  A/B Test   : {result['ab_id']}")
        print(f"  Audio File : {result['demo_wav']}")
        print(f"  Studio URL : {result['ui_url']}")
        print("Start the studio server with: uv run ambient-sound-ui")
        return


if __name__ == "__main__":
    main()
