from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ct_rtstruct_matching import find_ct_and_rtstruct
from editor_state import EditorState
from export_service import ExportService
from gui.slice_canvas import SliceCanvas
from study_loader import load_study
from viewer_state import ViewerState


class MainWindow(QMainWindow):
    """Focused glioma contour viewer/editor with uncertainty overlays."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Glioma RT Contour Lab")
        self.state = ViewerState()
        self.editor = EditorState()
        self.study = None
        self.unc_results: dict[str, dict] = {}
        self.edited_masks: dict[str, np.ndarray] = {}
        self._applying_brush = False
        self._busy_depth = 0

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(12)

        title = QLabel("Glioma Contour Lab")
        title.setObjectName("title")
        subtitle = QLabel("CT/MR review, AI-model contour preview, and contour editing")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)

        open_btn = QPushButton("Open Patient Folder")
        open_btn.clicked.connect(self._open_patient_folder)
        sidebar_layout.addWidget(open_btn)

        self.path_label = QLabel("No patient loaded")
        self.path_label.setWordWrap(True)
        sidebar_layout.addWidget(self.path_label)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        sidebar_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        sidebar_layout.addWidget(self.progress_bar)

        series_form = QFormLayout()
        self.base_series_combo = QComboBox()
        self.base_series_combo.currentTextChanged.connect(self._on_base_series_changed)
        series_form.addRow("Base image", self.base_series_combo)
        self.blend_series_combo = QComboBox()
        self.blend_series_combo.currentTextChanged.connect(self._on_blend_series_changed)
        series_form.addRow("Blend image", self.blend_series_combo)
        self.blend_opacity = QSlider(Qt.Orientation.Horizontal)
        self.blend_opacity.setRange(0, 100)
        self.blend_opacity.setValue(35)
        self.blend_opacity.valueChanged.connect(self._on_blend_opacity_changed)
        series_form.addRow("Blend opacity", self.blend_opacity)
        sidebar_layout.addLayout(series_form)

        self.structure_combo = QComboBox()
        self.structure_combo.currentTextChanged.connect(self._on_roi_changed)
        sidebar_layout.addWidget(QLabel("Selected structure"))
        sidebar_layout.addWidget(self.structure_combo)
        structure_hint = QLabel("Use the selected structure for editing and as the reference for other actions.")
        structure_hint.setWordWrap(True)
        structure_hint.setObjectName("hint")
        sidebar_layout.addWidget(structure_hint)

        sidebar_layout.addWidget(QLabel("Displayed structures"))
        self.overlay_list = QListWidget()
        self.overlay_list.itemChanged.connect(lambda _item: self._refresh_canvas())
        sidebar_layout.addWidget(self.overlay_list)
        overlay_hint = QLabel("Check the contours you want to see on top of the current image.")
        overlay_hint.setWordWrap(True)
        overlay_hint.setObjectName("hint")
        sidebar_layout.addWidget(overlay_hint)

        sidebar_layout.addWidget(QLabel("AI-model contour"))
        ai_form = QFormLayout()
        self.ai_target_combo = QComboBox()
        ai_form.addRow("Target region", self.ai_target_combo)
        self.unc_mode = QComboBox()
        self.unc_mode.addItems(["None", "Continuous", "Binary", "Both"])
        self.unc_mode.setCurrentText("Both")
        self.unc_mode.currentTextChanged.connect(lambda _text: self._refresh_canvas())
        ai_form.addRow("Overlay view", self.unc_mode)
        sidebar_layout.addLayout(ai_form)
        ai_hint = QLabel("Generate a simulated AI contour for the chosen region, then inspect the uncertainty overlay.")
        ai_hint.setWordWrap(True)
        ai_hint.setObjectName("hint")
        sidebar_layout.addWidget(ai_hint)
        create_unc_btn = QPushButton("Generate AI Model Contour")
        create_unc_btn.clicked.connect(self._on_create_uncertainty)
        sidebar_layout.addWidget(create_unc_btn)

        edit_frame = QFrame()
        edit_form = QFormLayout(edit_frame)
        edit_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        edit_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        edit_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        edit_form.setHorizontalSpacing(12)
        edit_form.setVerticalSpacing(10)
        self.enable_edit_checkbox = QCheckBox("Enable brush editing")
        self.enable_edit_checkbox.toggled.connect(self._on_edit_toggled)
        edit_form.addRow(self.enable_edit_checkbox)
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItems(["add", "erase"])
        self.brush_mode_combo.currentTextChanged.connect(self._on_brush_mode_changed)
        edit_form.addRow("Brush mode", self.brush_mode_combo)
        self.brush_radius_spin = QSpinBox()
        self.brush_radius_spin.setRange(1, 50)
        self.brush_radius_spin.setValue(6)
        self.brush_radius_spin.valueChanged.connect(self._on_brush_radius_changed)
        edit_form.addRow("Brush radius", self.brush_radius_spin)

        edit_actions = QHBoxLayout()
        edit_actions.setSpacing(10)
        clone_btn = QPushButton("Create Editable Copy")
        clone_btn.clicked.connect(self._clone_active_roi_for_edit)
        clone_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        clone_btn.setMinimumHeight(38)
        edit_actions.addWidget(clone_btn)
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self._undo_edit)
        undo_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        undo_btn.setMinimumHeight(38)
        edit_actions.addWidget(undo_btn)
        edit_actions_widget = QWidget()
        edit_actions_widget.setLayout(edit_actions)
        edit_form.addRow(edit_actions_widget)

        save_btn = QPushButton("Save Edited RTSTRUCT")
        save_btn.clicked.connect(self._save_edited_roi)
        save_btn.setMinimumHeight(40)
        save_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        edit_form.addRow(save_btn)
        sidebar_layout.addWidget(edit_frame)
        sidebar_layout.addStretch(1)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setFixedWidth(336)
        sidebar_scroll.setWidget(sidebar)

        viewer = QFrame()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setSpacing(12)

        self.canvas = SliceCanvas()
        self.canvas.strokeStarted.connect(self._begin_brush_stroke)
        self.canvas.editRequested.connect(self._apply_brush)
        viewer_layout.addWidget(self.canvas, stretch=1)

        bottom_bar = QHBoxLayout()
        self.slice_label = QLabel("Slice 0")
        bottom_bar.addWidget(self.slice_label)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        bottom_bar.addWidget(self.slice_slider, stretch=1)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.clicked.connect(self._zoom_out)
        bottom_bar.addWidget(zoom_out_btn)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(self._zoom_in)
        bottom_bar.addWidget(zoom_in_btn)
        self.zoom_label = QLabel("100%")
        bottom_bar.addWidget(self.zoom_label)
        viewer_layout.addLayout(bottom_bar)

        main_layout.addWidget(sidebar_scroll)
        main_layout.addWidget(viewer, stretch=1)
        self.setCentralWidget(root)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f2f1ea;
                color: #1d2420;
                font-family: "DejaVu Sans";
                font-size: 13px;
            }
            QFrame#sidebar {
                background: #e2ddd0;
                border-radius: 18px;
                padding: 12px;
            }
            QScrollArea#sidebarScroll {
                background: transparent;
                border: none;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #4b564f;
            }
            QLabel#hint {
                color: #5f675f;
                font-size: 12px;
            }
            QPushButton {
                background: #1d2420;
                color: #f5f4ef;
                border-radius: 12px;
                padding: 10px 12px;
                border: none;
                min-height: 38px;
                text-align: center;
            }
            QPushButton:hover {
                background: #2f3a34;
            }
            QComboBox, QListWidget, QSpinBox, QSlider {
                background: #faf8f1;
                border: 1px solid #c1baab;
                border-radius: 10px;
                padding: 6px;
            }
            """
        )

    def _open_patient_folder(self) -> None:
        patient_dir_str = QFileDialog.getExistingDirectory(self, "Select patient/visit folder")
        if not patient_dir_str:
            return

        patient_dir = Path(patient_dir_str)
        try:
            self._set_busy(True, "Loading patient study...")
            match = find_ct_and_rtstruct(patient_dir, prefer_roi_substrings=["GTV", "CTV", "PTV"])
            self.study = load_study(match.ct_folder, match.rtstruct_path, eager_roi_masks=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Error",
                (
                    "Could not auto-match CT and RTSTRUCT from the selected folder.\n"
                    "The folder should contain a CT series and matching RTSTRUCT.\n\n"
                    f"Details: {exc}"
                ),
            )
            return
        finally:
            self._set_busy(False)

        self.unc_results.clear()
        self.edited_masks.clear()
        self.editor = EditorState()
        self.path_label.setText(
            f"Patient: {self.study.metadata.patient_id}\n"
            f"CT: {match.ct_folder.name}\n"
            f"RTSTRUCT: {match.rtstruct_path.name}\n"
            f"Images: {', '.join(self.study.image_series.keys())}"
        )

        self._populate_series_controls()
        self._populate_roi_controls()
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.study.ct_volume.shape[0] - 1)
        self.slice_slider.setValue(0)
        self.state.active_slice_index = 0
        self._refresh_canvas()

    def _populate_series_controls(self) -> None:
        if self.study is None:
            return
        names = list(self.study.image_series.keys())
        self.base_series_combo.blockSignals(True)
        self.blend_series_combo.blockSignals(True)
        self.base_series_combo.clear()
        self.blend_series_combo.clear()
        self.base_series_combo.addItems(names)
        self.blend_series_combo.addItem("None")
        for name in names:
            if name != "CT":
                self.blend_series_combo.addItem(name)
        self.base_series_combo.setCurrentText("CT")
        self.blend_series_combo.setCurrentText("None")
        self.base_series_combo.blockSignals(False)
        self.blend_series_combo.blockSignals(False)
        self.state.base_series = "CT"
        self.state.blend_series = None

    def _populate_roi_controls(self) -> None:
        if self.study is None:
            return
        self.structure_combo.blockSignals(True)
        self.structure_combo.clear()
        self.structure_combo.addItems(self.study.roi_names)
        self.structure_combo.blockSignals(False)
        self.state.active_roi = self.study.roi_names[0] if self.study.roi_names else None
        if self.state.active_roi:
            self.structure_combo.setCurrentText(self.state.active_roi)

        self.ai_target_combo.blockSignals(True)
        self.ai_target_combo.clear()
        self.ai_target_combo.addItems([name for name in self.study.roi_names if not name.startswith(("AI_", "Uncertainty_"))])
        self.ai_target_combo.blockSignals(False)
        if self.state.active_roi and self.state.active_roi in [self.ai_target_combo.itemText(i) for i in range(self.ai_target_combo.count())]:
            self.ai_target_combo.setCurrentText(self.state.active_roi)

        self.overlay_list.blockSignals(True)
        self.overlay_list.clear()
        for idx, roi_name in enumerate(self.study.roi_names):
            item = QListWidgetItem(roi_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if idx < 3 else Qt.Unchecked)
            self.overlay_list.addItem(item)
        self.overlay_list.blockSignals(False)

    def _on_base_series_changed(self, series_name: str) -> None:
        self.state.base_series = series_name or "CT"
        self._refresh_canvas()

    def _on_blend_series_changed(self, series_name: str) -> None:
        self.state.blend_series = None if not series_name or series_name == "None" else series_name
        self._refresh_canvas()

    def _on_blend_opacity_changed(self, value: int) -> None:
        self.state.blend_opacity = value / 100.0
        self._refresh_canvas()

    def _on_roi_changed(self, roi_name: str) -> None:
        self.state.active_roi = roi_name if roi_name else None
        self.editor.active_roi = self.state.active_roi
        if roi_name in self.edited_masks:
            self.editor.edited_roi_name = roi_name
            self.editor.editable_mask_volume = self.edited_masks[roi_name]
        elif roi_name not in self.edited_masks:
            self.editor.edited_roi_name = None
            self.editor.editable_mask_volume = None
        if roi_name and roi_name in [self.ai_target_combo.itemText(i) for i in range(self.ai_target_combo.count())]:
            self.ai_target_combo.setCurrentText(roi_name)
        self._refresh_canvas()

    def _on_edit_toggled(self, checked: bool) -> None:
        self.editor.editing_enabled = checked
        self.canvas.set_editing(checked, self.editor.brush_mode)
        if checked and self.editor.editable_mask_volume is None:
            self._clone_active_roi_for_edit()
        self._refresh_canvas()

    def _on_brush_mode_changed(self, mode: str) -> None:
        self.editor.brush_mode = mode
        self.canvas.set_editing(self.editor.editing_enabled, mode)

    def _on_brush_radius_changed(self, value: int) -> None:
        self.editor.brush_radius = int(value)

    def _clone_active_roi_for_edit(self) -> None:
        if self.study is None or not self.state.active_roi:
            return
        source_name = self.state.active_roi
        source_mask = self.study.roi_masks.get(source_name)
        if source_mask is None:
            return
        edited_name = f"{source_name}_edited"
        editable = source_mask.copy()
        self.edited_masks[edited_name] = editable
        self.study.roi_masks[edited_name] = editable
        if edited_name not in self.study.roi_names:
            self.study.roi_names.append(edited_name)
            self.structure_combo.addItem(edited_name)
            item = QListWidgetItem(edited_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.overlay_list.addItem(item)
        self.editor.active_roi = source_name
        self.editor.edited_roi_name = edited_name
        self.editor.editable_mask_volume = editable
        self.state.active_roi = edited_name
        self.structure_combo.setCurrentText(edited_name)
        self._refresh_canvas()

    def _undo_edit(self) -> None:
        if not self.editor.undo_stack or self.editor.edited_roi_name is None:
            return
        restored = self.editor.undo_stack.pop()
        self.edited_masks[self.editor.edited_roi_name] = restored
        self.study.roi_masks[self.editor.edited_roi_name] = restored
        self.editor.editable_mask_volume = restored
        self._refresh_canvas()

    def _save_edited_roi(self) -> None:
        if self.study is None or self.editor.edited_roi_name is None or self.editor.editable_mask_volume is None:
            QMessageBox.information(self, "No edited contour", "Create or modify an edited contour first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return

        try:
            self._set_busy(True, "Saving edited RTSTRUCT...")
            export_path = ExportService(Path(out_dir)).export_edited_roi(
                self.study,
                self.editor.edited_roi_name,
                self.editor.editable_mask_volume,
            )
        finally:
            self._set_busy(False)
        QMessageBox.information(self, "Export complete", f"Saved edited RTSTRUCT to:\n{export_path}")

    def _on_create_uncertainty(self) -> None:
        if self.study is None:
            QMessageBox.information(self, "No study", "Load a patient first.")
            return
        roi_name = self.ai_target_combo.currentText().strip()
        if not roi_name:
            QMessageBox.information(self, "No target region", "Choose the region for AI contour generation first.")
            return

        gt_mask = self.study.roi_masks.get(roi_name)
        if gt_mask is None:
            QMessageBox.critical(self, "Error", f"ROI data not found: {roi_name}")
            return

        try:
            self._set_busy(True, f"Generating AI-model contour for {roi_name}...")
            from main import compute_surface_distance_pmap
            from sdf_perturb import PerturbParams, perturb_mask_via_sdf
            from unc_masks import make_uncertainty_outputs

            dx, dy, dz = self.study.geometry.spacing
            spacing_zyx = (float(dz), float(dy), float(dx))
            params = PerturbParams()
            mask_ai, _, _ = perturb_mask_via_sdf(gt_mask.copy(), spacing_zyx, params, seed=0, postprocess=True)
            pmap = compute_surface_distance_pmap(mask_ai, gt_mask, spacing_zyx)
            unc = make_uncertainty_outputs(
                pmap,
                mask_ai,
                heat_method="passthrough",
                contour_mode="top_fraction",
                contour_top_fraction=0.25,
            )

            unc_name = f"AI_{roi_name}"
            # The AI layer should display the perturbed contour itself so the
            # visible blob matches the uncertainty map derived from that mask.
            self.study.roi_masks[unc_name] = mask_ai.astype(bool).copy()
            if unc_name not in self.study.roi_names:
                self.study.roi_names.append(unc_name)
                self.structure_combo.addItem(unc_name)
                item = QListWidgetItem(unc_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.overlay_list.addItem(item)
            self.unc_results[unc_name] = {"mask_ai": mask_ai, "pmap": pmap, "unc": unc, "blob_heat": unc.blob_heat}
            self.structure_combo.setCurrentText(unc_name)
            self._refresh_canvas()
        except Exception as exc:
            QMessageBox.critical(self, "Uncertainty Error", f"Failed to create uncertainty: {exc}")
        finally:
            self._set_busy(False)

    def _zoom_in(self) -> None:
        self.state.zoom = min(6.0, self.state.zoom * 1.2)
        self._update_zoom_label()
        self._refresh_canvas()

    def _zoom_out(self) -> None:
        self.state.zoom = max(0.2, self.state.zoom / 1.2)
        self._update_zoom_label()
        self._refresh_canvas()

    def _update_zoom_label(self) -> None:
        self.zoom_label.setText(f"{int(round(self.state.zoom * 100))}%")

    def _on_slice_changed(self, index: int) -> None:
        self.state.active_slice_index = index
        self.editor.active_slice_index = index
        self.slice_label.setText(f"Slice {index}")
        self._refresh_canvas()

    def _visible_roi_names(self) -> list[str]:
        visible: list[str] = []
        for row in range(self.overlay_list.count()):
            item = self.overlay_list.item(row)
            if item.checkState() == Qt.Checked:
                visible.append(item.text())
        return visible

    def _get_display_slices(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.study is None:
            raise RuntimeError("No study loaded")
        z = self.state.active_slice_index
        base_series = self.study.image_series[self.state.base_series]
        base_slice = base_series.volume[z]
        blend_slice = None
        if self.state.blend_series:
            blend_series = self.study.image_series.get(self.state.blend_series)
            if blend_series is not None and blend_series.volume.shape == base_series.volume.shape:
                blend_slice = blend_series.volume[z]
        return base_slice, blend_slice

    def _apply_brush(self, x: int, y: int, mode: str) -> None:
        if self.study is None or not self.editor.editing_enabled:
            return
        if self.editor.edited_roi_name is None or self.editor.editable_mask_volume is None:
            return
        if self._applying_brush:
            return

        self._applying_brush = True
        try:
            volume = self.editor.editable_mask_volume
            z = self.state.active_slice_index
            current_slice = volume[z]
            yy, xx = np.ogrid[: current_slice.shape[0], : current_slice.shape[1]]
            radius = self.editor.brush_radius
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            if mode == "erase":
                current_slice[mask] = False
            else:
                current_slice[mask] = True
            volume[z] = current_slice
            self.edited_masks[self.editor.edited_roi_name] = volume
            self.study.roi_masks[self.editor.edited_roi_name] = volume
            self._refresh_canvas()
        finally:
            self._applying_brush = False

    def _begin_brush_stroke(self) -> None:
        if self.editor.editable_mask_volume is None:
            return
        self.editor.undo_stack.append(self.editor.editable_mask_volume.copy())

    def _refresh_canvas(self) -> None:
        if self.study is None:
            return

        z = self.state.active_slice_index
        self.slice_label.setText(f"Slice {z}")
        base_slice, blend_slice = self._get_display_slices()

        roi_layers: list[tuple[str, np.ndarray]] = []
        for roi_name in self._visible_roi_names():
            mask = self.study.roi_masks.get(roi_name)
            if mask is not None:
                roi_layers.append((roi_name, mask[z]))

        unc_heat = None
        contour_bin = None
        active_roi = self.state.active_roi
        if active_roi in self.unc_results:
            entry = self.unc_results[active_roi]
            blob = entry.get("blob_heat")
            pmap = entry.get("pmap")
            unc = entry.get("unc")
            if self.unc_mode.currentText() in {"Continuous", "Both"}:
                unc_heat = blob[z] if blob is not None else pmap[z]
            if self.unc_mode.currentText() in {"Binary", "Both"} and unc is not None:
                contour_bin = unc.contour_bin[z].astype(bool)

        active_edit_slice = None
        if (
            self.editor.editing_enabled
            and self.editor.edited_roi_name
            and self.editor.editable_mask_volume is not None
            and self.editor.edited_roi_name in self._visible_roi_names()
        ):
            active_edit_slice = self.editor.editable_mask_volume[z]

        self.canvas.set_editing(self.editor.editing_enabled, self.editor.brush_mode)
        self.canvas.render_slice(
            base_slice,
            blend_slice=blend_slice,
            blend_opacity=self.state.blend_opacity,
            roi_layers=roi_layers,
            active_edit_mask=active_edit_slice,
            unc_heat=unc_heat,
            contour_bin=contour_bin,
            zoom=self.state.zoom,
        )

    def _set_busy(self, busy: bool, message: str = "") -> None:
        if busy:
            self._busy_depth += 1
            self.progress_label.setText(message)
            self.progress_bar.show()
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            self._busy_depth = max(0, self._busy_depth - 1)
            if self._busy_depth == 0:
                self.progress_label.clear()
                self.progress_bar.hide()
                QApplication.restoreOverrideCursor()
        QApplication.processEvents()
