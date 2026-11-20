"""Repository license, dependency, and clean-room hygiene verification."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROHIBITED_PACKAGES = {
    "pedalboard",
    "matchering",
    "pyrubberband",
    "essentia",
    "phantom",
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".aif"}
MAX_AUDIO_BYTES = 1_000_000


def main() -> int:
    errors: list[str] = []

    for req_file in ("LICENSE", "NOTICE", "pyproject.toml", "README.md"):
        if not (ROOT / req_file).exists():
            errors.append(f"Missing required root file: {req_file}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    if project.get("license") != "Apache-2.0":
        errors.append(
            f"pyproject.toml license must be 'Apache-2.0', got {project.get('license')!r}"
        )

    deps: list[str] = list(project.get("dependencies", []))
    for group_deps in pyproject.get("dependency-groups", {}).values():
        deps.extend(group_deps)
    for dep in deps:
        pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower()
        if pkg_name in PROHIBITED_PACKAGES:
            errors.append(f"Prohibited copyleft/incompatible dependency found: {dep}")

    # Check git-tracked files when inside a git repository
    if (ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for rel_path in proc.stdout.splitlines():
                norm = rel_path.replace("\\", "/")
                if norm == ".local" or norm.startswith(".local/"):
                    errors.append(f"Private '.local/' path must not be git-tracked: {norm}")
                if norm.startswith(".ambientsound/"):
                    errors.append(
                        f"Runtime workspace '.ambientsound/' must not be git-tracked: {norm}"
                    )
                full = ROOT / rel_path
                if full.exists() and full.suffix.lower() in AUDIO_EXTENSIONS:
                    size = full.stat().st_size
                    if size > MAX_AUDIO_BYTES:
                        errors.append(
                            f"Large audio file ({size} bytes) must not be git-tracked: {norm}"
                        )

    if errors:
        for err in errors:
            print(f"[HYGIENE ERROR] {err}", file=sys.stderr)
        return 1

    print("Hygiene check passed: Apache-2.0 license, clean dependencies, and safe tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
