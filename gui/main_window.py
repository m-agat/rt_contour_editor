from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from gui.slice_canvas import SliceCanvas
from study_loader import load_study
from viewer_state import ViewerState


class MainWindow(QMainWindow):
    """Minimal MVP viewer for CT + RTSTRUCT ROI overlays."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RT Contour Editor MVP")
        self.state = ViewerState()
        self.study = None

        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QHBoxLayout()
        open_btn = QPushButton("Open Study")
        open_btn.clicked.connect(self._open_study)
        controls.addWidget(open_btn)

        self.roi_combo = QComboBox()
        self.roi_combo.currentTextChanged.connect(self._on_roi_changed)
        controls.addWidget(QLabel("ROI:"))
        controls.addWidget(self.roi_combo)
        layout.addLayout(controls)

        self.canvas = SliceCanvas()
        layout.addWidget(self.canvas)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.slice_slider)

        self.setCentralWidget(root)

    def _open_study(self) -> None:
        ct_dir = QFileDialog.getExistingDirectory(self, "Select CT folder")
        if not ct_dir:
            return
        rtstruct_path, _ = QFileDialog.getOpenFileName(self, "Select RTSTRUCT", filter="DICOM (*.dcm);;All files (*)")
        if not rtstruct_path:
            return

        self.study = load_study(Path(ct_dir), Path(rtstruct_path), eager_roi_masks=True)
        self.roi_combo.clear()
        self.roi_combo.addItems(self.study.roi_names)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.study.ct_volume.shape[0] - 1)
        self.state.active_slice_index = 0
        if self.study.roi_names:
            self.state.active_roi = self.study.roi_names[0]
        self._refresh_canvas()

    def _on_roi_changed(self, roi_name: str) -> None:
        self.state.active_roi = roi_name if roi_name else None
        self._refresh_canvas()

    def _on_slice_changed(self, index: int) -> None:
        self.state.active_slice_index = index
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        if self.study is None:
            return
        z = self.state.active_slice_index
        ct_slice = self.study.ct_volume[z]
        roi_mask = None
        if self.state.active_roi:
            roi_volume = self.study.roi_masks.get(self.state.active_roi)
            if roi_volume is not None:
                roi_mask = roi_volume[z]
        self.canvas.render_slice(ct_slice, roi_mask)
