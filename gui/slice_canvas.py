from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap, QColor, QBrush
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
        # Hover label state (shows ROI/AI/uncertainty at cursor)
        self._hover_labels_enabled = True
        self._last_mouse_pos: QPoint | None = None
        # store last rendered overlay slices for querying by hover
        self._last_roi_layers: list[tuple[str, np.ndarray]] = []
        self._last_unc_heat_slice: np.ndarray | None = None
        self._last_contour_bin: np.ndarray | None = None
        # tooltip explaining uncertainty
        self.setToolTip("Uncertainty: High uncertainty indicates model disagreement near the boundary.")

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
        original_edit_mask: np.ndarray | None = None,
        unc_heat: np.ndarray | None = None,
        contour_bin: np.ndarray | None = None,
        zoom: float = 1.0,
        active_roi: str | None = None,
        unc_mode_text: str | None = None,
    ) -> None:
        self._zoom = float(max(zoom, 0.1))
        self._image_shape = base_slice.shape
        rgb = self._render_rgb(
            base_slice=base_slice,
            blend_slice=blend_slice,
            blend_opacity=blend_opacity,
            roi_layers=roi_layers or [],
            active_edit_mask=active_edit_mask,
            original_edit_mask=original_edit_mask,
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
        # store info for hover queries
        self._last_roi_layers = [(n, np.asarray(m, dtype=bool)) for n, m in (roi_layers or [])]
        self._last_unc_heat_slice = np.asarray(unc_heat, dtype=float) if unc_heat is not None else None
        self._last_contour_bin = np.asarray(contour_bin, dtype=bool) if contour_bin is not None else None
        # keep active labels info too
        self._last_active_roi = active_roi
        self._last_unc_mode = unc_mode_text

    def set_hover_labels_enabled(self, show: bool) -> None:
        self._hover_labels_enabled = bool(show)
        self.update()

    def set_legend_visibility(self, show: bool) -> None:
        self._show_legend = bool(show)
        self.update()

    def _render_rgb(
        self,
        *,
        base_slice: np.ndarray,
        blend_slice: np.ndarray | None,
        blend_opacity: float,
        roi_layers: list[tuple[str, np.ndarray]],
        active_edit_mask: np.ndarray | None,
        original_edit_mask: np.ndarray | None,
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

        # Show edited mask with subtle overlay and highlight differences vs original
        if active_edit_mask is not None:
            edit_mask = np.asarray(active_edit_mask, dtype=bool)
            # base edit overlay (cyan-ish)
            edit_color = np.array([0, 229, 255], dtype=np.float32)
            edit_alpha = 0.28
            rgb_f[edit_mask] = (1.0 - edit_alpha) * rgb_f[edit_mask] + edit_alpha * edit_color

            # If we have the original mask, highlight additions/removals
            if original_edit_mask is not None:
                orig_mask = np.asarray(original_edit_mask, dtype=bool)
                added = edit_mask & (~orig_mask)
                removed = orig_mask & (~edit_mask)
                if added.any():
                    add_col = np.array([0, 200, 0], dtype=np.float32)  # green for added
                    add_alpha = 0.6
                    rgb_f[added] = (1.0 - add_alpha) * rgb_f[added] + add_alpha * add_col
                if removed.any():
                    rem_col = np.array([255, 50, 50], dtype=np.float32)  # red for removed
                    rem_alpha = 0.6
                    rgb_f[removed] = (1.0 - rem_alpha) * rgb_f[removed] + rem_alpha * rem_col

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

    def enterEvent(self, event) -> None:  # show hover when entering
        super().enterEvent(event)
        self._last_mouse_pos = None

    def leaveEvent(self, event) -> None:  # hide hover when leaving
        super().leaveEvent(event)
        self._last_mouse_pos = None
        self.update()

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
        # track mouse for hover labels regardless of edit state
        try:
            self._last_mouse_pos = event.position().toPoint()
        except Exception:
            self._last_mouse_pos = event.pos()
        self.update()

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
        painter = QPainter(self)

        # Draw edit mode label when editing enabled
        if self._edit_enabled:
            pen = QPen(Qt.white)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawText(12, 24, "Edit mode")

        # Hover labels: show ROI names / AI / uncertainty under cursor
        if self._hover_labels_enabled and self._last_mouse_pos is not None:
            pixmap = self.pixmap()
            if pixmap is not None and not pixmap.isNull():
                coords = self._event_to_image_xy(self._last_mouse_pos)
                if coords is not None:
                    img_x, img_y = coords
                    labels: list[str] = []
                    # check ROI layers
                    for name, mask in self._last_roi_layers:
                        try:
                            if mask[img_y, img_x]:
                                labels.append(name)
                        except Exception:
                            continue
                    # contour bin
                    try:
                        if self._last_contour_bin is not None and self._last_contour_bin[img_y, img_x]:
                            labels.append("contour")
                    except Exception:
                        pass
                    # uncertainty value
                    unc_text = None
                    try:
                        if self._last_unc_heat_slice is not None:
                            val = float(self._last_unc_heat_slice[img_y, img_x])
                            unc_text = f"uncertainty: {val:.3f}"
                    except Exception:
                        unc_text = None
                    if unc_text:
                        labels.append(unc_text)

                    if labels:
                        text = ", ".join(labels)
                        fm = painter.fontMetrics()
                        padding = 6
                        text_w = fm.horizontalAdvance(text)
                        text_h = fm.height()
                        # draw near cursor but keep inside pixmap bounds
                        w = self.width()
                        h = self.height()
                        px = self._last_mouse_pos.x()
                        py = self._last_mouse_pos.y()
                        box_w = text_w + padding * 2
                        box_h = text_h + padding * 2
                        box_x = px + 12
                        box_y = py + 12
                        if box_x + box_w > w:
                            box_x = px - box_w - 12
                        if box_y + box_h > h:
                            box_y = py - box_h - 12

                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 6, 6)
                        painter.setPen(Qt.white)
                        painter.drawText(box_x + padding, box_y + padding + fm.ascent(), text)

        painter.end()
