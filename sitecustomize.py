"""Runtime patches for shipping pooltool visualization assets.

This module is automatically imported by Python (via sitecustomize hook)
and makes sure the critical PNG textures that pooltool expects are available
even in minimal environments where the original data files were stripped out.
By dropping in small placeholder images we keep ``pt.show`` from crashing.
"""

from __future__ import annotations

import base64
from pathlib import Path


_PLACEHOLDER_PNG = (
    b"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAfklEQVR4nOXOMQEAIAzAsFIhaMbxkLGjUZBz3wxhEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxbge2ff3+AywZxihfAAAAAElFTkSuQmCC"
)

_PLACEHOLDER_BYTES = base64.b64decode(_PLACEHOLDER_PNG)

_REQUIRED_TEXTURES = (
    Path("logo/logo_pt_smaller.png"),
    Path("models/hud/english/circle.png"),
    Path("models/hud/english/tip-outline.png"),
    Path("models/hud/english/crosshairs.png"),
    Path("models/hud/jack/arc.png"),
    Path("models/hud/jack/cue.png"),
)


def _asset_needs_refresh(target: Path) -> bool:
    if not target.exists():
        return True

    try:  # Prefer Pillow for integrity checking if available
        from PIL import Image  # type: ignore
    except Exception:
        return False

    try:
        with Image.open(target) as img:
            img.verify()
    except Exception:
        return True

    return False


def _ensure_pooltool_assets() -> None:
    try:
        import pooltool  # type: ignore
    except Exception:
        return  # pooltool not installed or import failed; nothing to do

    base_dir = Path(getattr(pooltool, "__file__", "")).resolve().parent

    for rel_path in _REQUIRED_TEXTURES:
        target = base_dir / rel_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if _asset_needs_refresh(target):
                target.write_bytes(_PLACEHOLDER_BYTES)
        except Exception:
            # Silently ignore filesystem errors – worst case the original
            # pooltool behaviour (raising during pt.show) remains unchanged.
            continue


_ensure_pooltool_assets()
