# Contributing to AmbientSound

Thank you for contributing to **AmbientSound**!

## License & Contributions
AmbientSound is licensed under the **Apache License, Version 2.0**. By submitting a contribution (pull request, patch, or issue snippet), you agree that your contribution is licensed under the Apache-2.0 terms (see Section 5 of [LICENSE](LICENSE)).

## Dependency & Clean-Room Policy
1. **Permissive Dependencies Only:** All core runtime and dev dependencies must be compatible with Apache-2.0 (`Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `PSF-2.0`, `MPL-2.0`). Do **not** introduce GPL or AGPL Python/Rust/C dependencies into the core package.
2. **External Subprocess Tools:** Calling user-installed external binaries such as `ffmpeg` over subprocesses stays separate from core package linking, while keeping AmbientSound itself Apache-2.0.
3. **Clean-Room Engineering:** Contributions must be original code or adapted solely from permissively licensed open-source works with proper attribution in `NOTICE`. Do not submit code derived from reverse-engineering proprietary software.
4. **Model Weights:** Any optional neural weights with non-commercial or field-of-use restrictions must never be bundled in the repository and must require explicit opt-in via `AMBIENTSOUND_ALLOW_NONCOMMERCIAL=1`.

## Development Setup

```bash
uv sync --python 3.12 --all-groups
uv run ruff check .
uv run pytest -v
uv run python scripts/check_hygiene.py
```

## Pull Request Checklist
- [ ] All tests pass (`uv run pytest -v`) and new DSP/MCP behavior includes unit or integration tests.
- [ ] Linter passes (`uv run ruff check .`).
- [ ] Hygiene check passes (`uv run python scripts/check_hygiene.py`).
- [ ] No audio files or `.local/` private files are staged.
- [ ] Updated `CHANGELOG.md` (`[Unreleased]`) and relevant files in `docs/` if tool schemas or architecture changed.
