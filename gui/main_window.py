from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ct_rtstruct_matching import find_ct_and_rtstruct
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
        open_btn = QPushButton("Open Patient Folder")
        open_btn.clicked.connect(self._open_patient_folder)
        controls.addWidget(open_btn)

        self.roi_combo = QComboBox()
        self.roi_combo.currentTextChanged.connect(self._on_roi_changed)
        controls.addWidget(QLabel("ROI:"))
        controls.addWidget(self.roi_combo)
        # Uncertainty display mode selector
        self.unc_mode = QComboBox()
        self.unc_mode.addItems(["None", "Continuous", "Binary", "Both"])
        self.unc_mode.setCurrentText("Both")
        self.unc_mode.setToolTip("Select which uncertainty overlay to display")
        controls.addWidget(QLabel("Uncertainty:"))
        controls.addWidget(self.unc_mode)
        create_unc_btn = QPushButton("Create Uncertainty")
        create_unc_btn.clicked.connect(self._on_create_uncertainty)
        controls.addWidget(create_unc_btn)
        # Zoom controls
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.clicked.connect(self._zoom_in)
        controls.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.clicked.connect(self._zoom_out)
        controls.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        controls.addWidget(self.zoom_label)
        layout.addLayout(controls)

        self.path_label = QLabel("No patient loaded")
        layout.addWidget(self.path_label)

        self.canvas = SliceCanvas()
        layout.addWidget(self.canvas)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.slice_slider)

        self.setCentralWidget(root)

    def _open_patient_folder(self) -> None:
        patient_dir_str = QFileDialog.getExistingDirectory(self, "Select patient/visit folder")
        if not patient_dir_str:
            return

        patient_dir = Path(patient_dir_str)
        try:
            match = find_ct_and_rtstruct(patient_dir, prefer_roi_substrings=["GTV", "PTV"])
            self.study = load_study(match.ct_folder, match.rtstruct_path, eager_roi_masks=True)
            self.path_label.setText(
                f"Patient folder: {patient_dir}\nCT: {match.ct_folder.name} | RTSTRUCT: {match.rtstruct_path.name}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Error",
                (
                    "Could not auto-match CT + RTSTRUCT from the selected folder.\n"
                    "Please select a folder that contains both a CT series and matching RTSTRUCT.\n\n"
                    f"Details: {exc}"
                ),
            )
            return

        self.roi_combo.clear()
        self.roi_combo.addItems(self.study.roi_names)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.study.ct_volume.shape[0] - 1)
        self.state.active_slice_index = 0
        self.state.active_roi = self.study.roi_names[0] if self.study.roi_names else None
        self._refresh_canvas()

    def _on_roi_changed(self, roi_name: str) -> None:
        self.state.active_roi = roi_name if roi_name else None
        self._refresh_canvas()

    def _on_create_uncertainty(self) -> None:
        """Generate a perturbed AI mask and uncertainty maps for the active ROI.

        This uses the same SDF perturbation and distance-to-surface p-map used
        in the batch pipeline. Results are cached per-ROI in memory and shown
        on the canvas immediately.
        """
        if self.study is None:
            QMessageBox.information(self, "No study", "Load a patient first.")
            return
        if not self.state.active_roi:
            QMessageBox.information(self, "No ROI", "Select an ROI first.")
            return

        roi_name = self.state.active_roi
        gt_mask = self.study.roi_masks.get(roi_name)
        # Defensive copy: work on a copy so we never mutate the in-memory
        # study ROI arrays that are considered the single source of truth.
        if gt_mask is not None:
            gt_mask = gt_mask.copy()
        if gt_mask is None:
            QMessageBox.critical(self, "Error", f"ROI data not found: {roi_name}")
            return

        try:
            # Lazily import the perturbation + pmap helpers
            from sdf_perturb import PerturbParams, perturb_mask_via_sdf
            from main import compute_surface_distance_pmap
            from unc_masks import make_uncertainty_outputs
            from scipy.ndimage import binary_dilation, generate_binary_structure
            from main import AI_NEIGHBOR_DILATE

            # Convert spacing (dx, dy, dz) -> (dz, dy, dx) expected by SDF functions
            dx, dy, dz = self.study.geometry.spacing
            spacing_zyx = (float(dz), float(dy), float(dx))

            params = PerturbParams()
            mask_ai, sdf_mm, delta_mm = perturb_mask_via_sdf(gt_mask, spacing_zyx, params, seed=0, postprocess=True)

            pmap = compute_surface_distance_pmap(mask_ai, gt_mask, spacing_zyx)

            unc = make_uncertainty_outputs(
                pmap, mask_ai,
                heat_method="passthrough",
                contour_mode="top_fraction",
                contour_top_fraction=0.25,
            )

            # Create an uncertainty ROI that preserves the original tissue
            # shape (so the selected tissue remains intact). The per-voxel
            # heatmap `blob_heat` (computed by make_uncertainty_outputs) will
            # be used to visualize darker=more uncertain inside that ROI.
            unc_name = f"Uncertainty_{roi_name}"
            # Use a copy of the original GT mask (preserve tissue shape)
            gt_mask_bool = gt_mask.astype(bool).copy()
            if unc_name in self.study.roi_names:
                self.study.roi_masks[unc_name] = gt_mask_bool
            else:
                self.study.roi_names.append(unc_name)
                self.study.roi_masks[unc_name] = gt_mask_bool
                self.roi_combo.addItem(unc_name)

            # Cache results on the window for display under the uncertainty ROI
            # name only. Do NOT overwrite or alias the original ROI name so the
            # GTV/FLAIR entry continues to show the original segmentation.
            if not hasattr(self, "unc_results"):
                self.unc_results = {}
            entry = dict(mask_ai=mask_ai, pmap=pmap, unc=unc, blob_heat=unc.blob_heat)
            self.unc_results[unc_name] = entry

            QMessageBox.information(self, "Uncertainty", f"Uncertainty generated for ROI '{roi_name}'. Select '{unc_name}' from the ROI menu to view filled uncertainty.")
            self._refresh_canvas()

        except Exception as exc:
            QMessageBox.critical(self, "Uncertainty Error", f"Failed to create uncertainty: {exc}")

    def _zoom_in(self) -> None:
        """Increase zoom (multiplicative)."""
        self.state.zoom = min(5.0, self.state.zoom * 1.2)
        self._update_zoom_label()
        self._refresh_canvas()

    def _zoom_out(self) -> None:
        """Decrease zoom (multiplicative)."""
        self.state.zoom = max(0.2, self.state.zoom / 1.2)
        self._update_zoom_label()
        self._refresh_canvas()

    def _update_zoom_label(self) -> None:
        pct = int(round(self.state.zoom * 100))
        self.zoom_label.setText(f"{pct}%")

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
        # Pass any generated uncertainty overlays (if available)
        unc_heat = None
        contour_bin = None
        if hasattr(self, "unc_results") and self.state.active_roi in getattr(self, "unc_results", {}):
            entry = self.unc_results[self.state.active_roi]
            # Prefer propagated blob heat (filled ROI) if available
            blob = entry.get("blob_heat")
            pmap = entry.get("pmap")
            unc = entry.get("unc")
            if blob is not None:
                unc_heat = blob[z]
            elif pmap is not None:
                unc_heat = pmap[z]
            if unc is not None and getattr(unc, "contour_bin", None) is not None:
                contour_bin = unc.contour_bin[z].astype(bool)
        # Respect the user's choice for which uncertainty overlay to display
        mode = self.unc_mode.currentText() if hasattr(self, 'unc_mode') else "Both"
        show_heat = False
        show_bin = False
        if mode == "Continuous":
            show_heat = True
        elif mode == "Binary":
            show_bin = True
        elif mode == "Both":
            show_heat = True
            show_bin = True

        # If the user chooses Binary-only display, do not show the GT (roi_mask)
        # overlay to avoid visual clutter; in all other modes keep the GT visible.
        roi_mask_to_pass = roi_mask
        if show_bin and not show_heat:
            roi_mask_to_pass = None

        heat_arg = unc_heat if show_heat else None
        bin_arg = contour_bin if show_bin else None

        self.canvas.render_slice(ct_slice, roi_mask_to_pass, unc_heat=heat_arg, contour_bin=bin_arg, zoom=self.state.zoom)
