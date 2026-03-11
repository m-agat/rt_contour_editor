from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt


class SliceCanvas(QLabel):
    """Simple QLabel-backed canvas for CT slice + ROI mask / uncertainty overlay rendering."""

    def render_slice(
        self,
        ct_slice: np.ndarray,
        roi_mask: np.ndarray | None = None,
        unc_heat: np.ndarray | None = None,
        contour_bin: np.ndarray | None = None,
        zoom: float = 1.0,
    ) -> None:
        ct = ct_slice.astype(np.float32)
        lo, hi = np.percentile(ct, [2, 98])
        ct = np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = np.repeat((ct[..., None] * 255).astype(np.uint8), 3, axis=2)

        h, w = ct.shape

        # Render uncertainty heatmap as a filled region (blob) when available.
        # We map uncertainty values to colors using the 'magma' colormap but
        # invert the mapping so that HIGH uncertainty -> darker colors and
        # LOW uncertainty -> brighter colors (as requested).
        if unc_heat is not None:
            heat = np.clip(np.asarray(unc_heat, dtype=np.float32), 0.0, 1.0)

            # Determine where to apply the heat: prefer ROI area so the heat
            # appears as a filled blob matching the segmentation. If ROI is
            # not available, fall back to contour_bin or any non-zero heat.
            if roi_mask is not None:
                heat_mask = roi_mask.astype(bool)
            elif contour_bin is not None:
                heat_mask = contour_bin.astype(bool)
            else:
                heat_mask = heat > 0

            # Use magma and invert values so 1.0 -> dark, 0.0 -> bright
            cmap = cm.get_cmap("magma")
            color_map = cmap(1.0 - heat)  # (h, w, 4)
            color_rgb = color_map[..., :3].astype(np.float32)

            # Alpha proportional to heat so high-uncertainty areas appear
            # more opaque. Zero outside the heat_mask.
            alpha = (0.85 * heat)[..., None]
            alpha[~heat_mask] = 0.0

            # Blend overlay
            rgb_f = rgb.astype(np.float32) / 255.0
            rgb_blend = (1.0 - alpha) * rgb_f + alpha * color_rgb
            rgb = np.clip((rgb_blend * 255.0), 0, 255).astype(np.uint8)
        else:
            # No heatmap supplied: fall back to simple green-filled ROI display
            if roi_mask is not None:
                mask = roi_mask.astype(bool)
                rgb[mask, 0] = 102
                rgb[mask, 1] = 255
                rgb[mask, 2] = 102

        # Overlay binary contour segments (solid orange)
        if contour_bin is not None:
            cbin = contour_bin.astype(bool)
            rgb[cbin, 0] = 255
            rgb[cbin, 1] = 165
            rgb[cbin, 2] = 0

        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(image.copy())
        if zoom != 1.0 and zoom > 0:
            tw = max(1, int(w * zoom))
            th = max(1, int(h * zoom))
            pix = pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)
