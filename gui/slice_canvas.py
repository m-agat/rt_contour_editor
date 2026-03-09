from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class SliceCanvas(QLabel):
    """Simple QLabel-backed canvas for CT slice + ROI mask overlay rendering."""

    def render_slice(self, ct_slice: np.ndarray, roi_mask: np.ndarray | None = None) -> None:
        ct = ct_slice.astype(np.float32)
        lo, hi = np.percentile(ct, [2, 98])
        ct = np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = np.repeat((ct[..., None] * 255).astype(np.uint8), 3, axis=2)

        if roi_mask is not None:
            mask = roi_mask.astype(bool)
            rgb[mask, 0] = 255
            rgb[mask, 1] = (rgb[mask, 1] * 0.3).astype(np.uint8)
            rgb[mask, 2] = (rgb[mask, 2] * 0.3).astype(np.uint8)

        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(image.copy()))
