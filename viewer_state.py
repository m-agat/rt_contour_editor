from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewerState:
    """Basic viewer state for slice navigation and ROI visibility."""

    active_roi: str | None = None
    active_slice_index: int = 0
    overlay_visible: bool = True
    zoom: float = 1.0
