from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel


ROI_COLORS = [
    np.array([102, 255, 102], dtype=np.uint8),
    np.array([66, 165, 245], dtype=np.uint8),
    np.array([255, 193, 7], dtype=np.uint8),
    np.array([244, 81, 30], dtype=np.uint8),
]


class SliceCanvas(QLabel):
    """Canvas for image blending, ROI overlays, and slice editing."""

    strokeStarted = Signal()
    strokeFinished = Signal()
    editRequested = Signal(int, int, str)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(512, 512)
        self.setMouseTracking(True)
        self._zoom = 1.0
        self._image_shape = (1, 1)
        self._edit_enabled = False
        self._brush_mode = "add"
        self._last_pixmap_size = (1, 1)

    def set_editing(self, enabled: bool, brush_mode: str) -> None:
        self._edit_enabled = enabled
        self._brush_mode = brush_mode
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def render_slice(
        self,
        base_slice: np.ndarray,
        *,
        blend_slice: np.ndarray | None = None,
        blend_opacity: float = 0.35,
        roi_layers: list[tuple[str, np.ndarray]] | None = None,
        active_edit_mask: np.ndarray | None = None,
        unc_heat: np.ndarray | None = None,
        contour_bin: np.ndarray | None = None,
        zoom: float = 1.0,
    ) -> None:
        self._zoom = float(max(zoom, 0.1))
        self._image_shape = base_slice.shape
        rgb = self._render_rgb(
            base_slice=base_slice,
            blend_slice=blend_slice,
            blend_opacity=blend_opacity,
            roi_layers=roi_layers or [],
            active_edit_mask=active_edit_mask,
            unc_heat=unc_heat,
            contour_bin=contour_bin,
        )

        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(image.copy())
        if self._zoom != 1.0:
            tw = max(1, int(w * self._zoom))
            th = max(1, int(h * self._zoom))
            pix = pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._last_pixmap_size = (pix.width(), pix.height())
        self.setPixmap(pix)

    def _render_rgb(
        self,
        *,
        base_slice: np.ndarray,
        blend_slice: np.ndarray | None,
        blend_opacity: float,
        roi_layers: list[tuple[str, np.ndarray]],
        active_edit_mask: np.ndarray | None,
        unc_heat: np.ndarray | None,
        contour_bin: np.ndarray | None,
    ) -> np.ndarray:
        ct = np.asarray(base_slice, dtype=np.float32)
        lo, hi = np.percentile(ct, [2, 98])
        ct = np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = np.repeat((ct[..., None] * 255).astype(np.uint8), 3, axis=2)

        if blend_slice is not None and blend_slice.shape == base_slice.shape:
            mr = np.asarray(blend_slice, dtype=np.float32)
            lo_mr, hi_mr = np.percentile(mr, [2, 98])
            mr = np.clip((mr - lo_mr) / max(hi_mr - lo_mr, 1e-6), 0.0, 1.0)
            mr_rgb = np.zeros_like(rgb, dtype=np.uint8)
            mr_rgb[..., 0] = (mr * 255).astype(np.uint8)
            mr_rgb[..., 1] = (mr * 120).astype(np.uint8)
            mr_rgb[..., 2] = (mr * 255).astype(np.uint8)
            alpha = float(np.clip(blend_opacity, 0.0, 1.0))
            rgb = np.clip((1.0 - alpha) * rgb + alpha * mr_rgb, 0, 255).astype(np.uint8)

        rgb_f = rgb.astype(np.float32)
        for idx, (_name, roi_mask) in enumerate(roi_layers):
            color = ROI_COLORS[idx % len(ROI_COLORS)].astype(np.float32)
            mask = np.asarray(roi_mask, dtype=bool)
            alpha = 0.28
            rgb_f[mask] = (1.0 - alpha) * rgb_f[mask] + alpha * color

        if unc_heat is not None:
            heat = np.clip(np.asarray(unc_heat, dtype=np.float32), 0.0, 1.0)
            cmap = cm.get_cmap("magma")
            color_map = cmap(1.0 - heat)
            color_rgb = (255.0 * color_map[..., :3]).astype(np.float32)
            heat_mask = heat > 0
            alpha = (0.7 * heat)[..., None]
            rgb_f[heat_mask] = (1.0 - alpha[heat_mask]) * rgb_f[heat_mask] + alpha[heat_mask] * color_rgb[heat_mask]

        if active_edit_mask is not None:
            edit_mask = np.asarray(active_edit_mask, dtype=bool)
            color = np.array([0, 229, 255], dtype=np.float32)
            alpha = 0.35
            rgb_f[edit_mask] = (1.0 - alpha) * rgb_f[edit_mask] + alpha * color

        rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        if contour_bin is not None:
            cbin = np.asarray(contour_bin, dtype=bool)
            rgb[cbin, 0] = 255
            rgb[cbin, 1] = 165
            rgb[cbin, 2] = 0

        return rgb

    def _event_to_image_xy(self, event_pos: QPoint) -> tuple[int, int] | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        label_w = self.width()
        label_h = self.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        offset_x = (label_w - pix_w) / 2.0
        offset_y = (label_h - pix_h) / 2.0
        x = event_pos.x() - offset_x
        y = event_pos.y() - offset_y
        if x < 0 or y < 0 or x >= pix_w or y >= pix_h:
            return None

        img_h, img_w = self._image_shape
        img_x = int(np.clip((x / pix_w) * img_w, 0, img_w - 1))
        img_y = int(np.clip((y / pix_h) * img_h, 0, img_h - 1))
        return img_x, img_y

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._edit_enabled and event.button() in {Qt.LeftButton, Qt.RightButton}:
            coords = self._event_to_image_xy(event.position().toPoint())
            if coords is not None:
                self.strokeStarted.emit()
                mode = self._brush_mode if event.button() == Qt.LeftButton else "erase"
                self.editRequested.emit(coords[0], coords[1], mode)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._edit_enabled and event.buttons() & (Qt.LeftButton | Qt.RightButton):
            coords = self._event_to_image_xy(event.position().toPoint())
            if coords is not None:
                mode = self._brush_mode if event.buttons() & Qt.LeftButton else "erase"
                self.editRequested.emit(coords[0], coords[1], mode)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._edit_enabled and event.button() in {Qt.LeftButton, Qt.RightButton}:
            self.strokeFinished.emit()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._edit_enabled:
            return
        painter = QPainter(self)
        pen = QPen(Qt.white)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawText(12, 24, "Edit mode")
        painter.end()
