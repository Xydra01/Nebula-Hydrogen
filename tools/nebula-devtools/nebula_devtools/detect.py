"""Language detection and worktree confinement for the build/test/lint tools.

Language is chosen by an explicit ``language`` argument when given, otherwise auto-detected from
the worktree (``Cargo.toml`` -> Rust, ``pyproject.toml`` -> Python). Confinement rejects any path
that resolves outside the worktree root or onto the retired drive (Requirement 3.5, 6.1, 6.2).
"""

from __future__ import annotations

import os
from pathlib import Path

#: The languages this server supports in v1.
SUPPORTED = ("python", "rust", "cpp")


class ConfinementError(Exception):
    """A requested path resolves outside the worktree root or onto the retired drive."""


class LanguageError(Exception):
    """The target language was not given and could not be determined, or is unsupported."""


def detect_language(cwd: str) -> str | None:
    """Detect the worktree's language from its manifest files.
    ``Cargo.toml`` -> ``"rust"``; ``pyproject.toml`` -> ``"python"``;
    ``CMakeLists.txt`` / ``compile_commands.json`` -> ``"cpp"``.
    Rust is checked first so a polyglot repo with multiple manifests resolves consistently; callers
    that need another language pass it explicitly. Returns ``None`` when none is present.
    """
    root = Path(cwd)
    if (root / "Cargo.toml").is_file():
        return "rust"
    if (root / "pyproject.toml").is_file():
        return "python"
    if (root / "CMakeLists.txt").is_file() or (root / "compile_commands.json").is_file():
        return "cpp"
    return None


def resolve_language(arg: str | None, cwd: str) -> str:
    """Resolve the target language: explicit ``arg`` wins, else detect from ``cwd``.

    Raises :class:`LanguageError` if an explicit arg is unsupported, or if no arg is given and the
    language cannot be determined (Requirement 3.5).
    """
    if arg is not None:
        lang = arg.strip().lower()
        if lang not in SUPPORTED:
            raise LanguageError(f"unsupported language {arg!r}; supported: {', '.join(SUPPORTED)}")
        return lang
    detected = detect_language(cwd)
    if detected is None:
        raise LanguageError(
            "could not determine language from the worktree "
            "(no Cargo.toml or pyproject.toml); pass an explicit `language`"
        )
    return detected


def _drive_of(path: Path) -> str | None:
    """The uppercased drive letter of an absolute path (for example ``C``), or ``None``."""
    drive = path.drive  # e.g. "C:"
    if len(drive) >= 2 and drive[1] == ":":
        return drive[0].upper()
    return None


def ensure_within_worktree(path: str, worktree_root: str, retired_drive: str = "C:") -> Path:
    """Return the resolved absolute ``path`` if it is inside ``worktree_root`` and off the retired
    drive; otherwise raise :class:`ConfinementError` (Requirement 6.2, 6.4).

    ``retired_drive`` is compared by drive letter, case-insensitively, mirroring the Rust resolver's
    retired-drive rule without reimplementing the full resolver.
    """
    root = Path(worktree_root).resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    retired_letter = retired_drive.strip().rstrip(":").upper()[:1] or "C"
    if _drive_of(target) == retired_letter:
        raise ConfinementError(f"path {target} is on the retired drive {retired_drive}")

    # Containment: `target` must be `root` or a descendant of it.
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ConfinementError(f"path {target} resolves outside the worktree root {root}") from e
    return target


def retired_drive_from_env() -> str:
    """The retired drive the host passes via ``NEBULA_RETIRED_DRIVE``, defaulting to ``C:``."""
    return os.environ.get("NEBULA_RETIRED_DRIVE", "C:")
