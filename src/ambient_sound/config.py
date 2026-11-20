"""Configuration and path-sandboxing helpers for AmbientSound."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime settings for AmbientSound workspaces, sandboxing, and UI."""

    workspace_dir: Path = field(default_factory=lambda: Path.cwd() / ".ambientsound")
    allowed_roots: list[Path] = field(default_factory=list)
    allow_noncommercial: bool = False
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir).expanduser().resolve()
        if not self.allowed_roots:
            self.allowed_roots = [Path.cwd().resolve(), self.workspace_dir]
        else:
            resolved_roots = [Path(r).expanduser().resolve() for r in self.allowed_roots]
            if self.workspace_dir not in resolved_roots:
                resolved_roots.append(self.workspace_dir)
            self.allowed_roots = resolved_roots

    @classmethod
    def from_env(cls) -> Settings:
        """Load configuration from environment variables with safe local defaults."""
        workspace_raw = os.environ.get("AMBIENTSOUND_WORKSPACE")
        workspace_dir = (
            Path(workspace_raw).expanduser().resolve()
            if workspace_raw
            else (Path.cwd() / ".ambientsound").resolve()
        )

        roots_raw = os.environ.get("AMBIENTSOUND_ALLOWED_ROOTS")
        if roots_raw:
            allowed_roots = [
                Path(part.strip()).expanduser().resolve()
                for part in roots_raw.split(os.pathsep)
                if part.strip()
            ]
        else:
            allowed_roots = [Path.cwd().resolve(), workspace_dir]

        ui_port_raw = os.environ.get("AMBIENTSOUND_UI_PORT")
        try:
            ui_port = int(ui_port_raw) if ui_port_raw else 8765
        except ValueError:
            ui_port = 8765

        return cls(
            workspace_dir=workspace_dir,
            allowed_roots=allowed_roots,
            allow_noncommercial=_parse_bool(
                os.environ.get("AMBIENTSOUND_ALLOW_NONCOMMERCIAL"), default=False
            ),
            ui_host=os.environ.get("AMBIENTSOUND_UI_HOST", "127.0.0.1"),
            ui_port=ui_port,
        )

    @property
    def ui_base_url(self) -> str:
        """Return base HTTP URL for the local A/B studio."""
        return f"http://{self.ui_host}:{self.ui_port}"


def ensure_path_allowed(path: str | Path, settings: Settings) -> Path:
    """Resolve a path and enforce that it stays within configured allowed_roots.

    Raises:
        PermissionError: If the resolved path is outside every allowed root.
    """
    candidate = Path(path).expanduser().resolve()
    for root in settings.allowed_roots:
        resolved_root = root.resolve()
        if candidate == resolved_root or candidate.is_relative_to(resolved_root):
            return candidate
    roots_display = ", ".join(str(r) for r in settings.allowed_roots)
    raise PermissionError(
        f"Path '{candidate}' is outside AMBIENTSOUND_ALLOWED_ROOTS ({roots_display})."
    )
