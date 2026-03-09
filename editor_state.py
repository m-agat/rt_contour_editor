from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EditorState:
    """Editing state scaffold for upcoming brush-based contour editing."""

    active_roi: str | None = None
    active_slice_index: int = 0
    editable_mask_volume: np.ndarray | None = None
    brush_mode: str = "add"
    brush_radius: int = 3
    undo_stack: list[np.ndarray] = field(default_factory=list)
