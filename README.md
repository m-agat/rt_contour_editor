# rt_contour_editor

## Desktop-app foundation architecture

This repository now includes a focused PySide6 glioma contour viewer/editor for research workflows while preserving existing extraction/export logic.

- `dicom_utils.py`: shared DICOM IO, CT discovery, orientation-aware slice sorting, spacing validation, and metadata extraction.
- `roi_extraction.py`: ROI extraction built on consistent geometry/slice ordering utilities.
- `models.py`: dataclass app models (`DicomStudyMetadata`, `RoiLayer`, `LoadedStudy`).
- `study_loader.py`: high-level `load_study(ct_folder, rtstruct_path)` API returning one coherent in-memory study, including optional MR discovery.
- `viewer_state.py` and `editor_state.py`: app/view/edit state models for slice navigation, image blending, and brush editing.
- `export_service.py`: app service wrapper to export edited ROI masks using existing `dicom_export.py` logic.
- `gui/`: PySide6 UI for loading a patient folder, CT/MR viewing, multiple ROI overlays, uncertainty display, brush editing, and RTSTRUCT export.
- `app.py`: desktop entrypoint.

## Implemented workflow

The current app supports:

- viewing CT with optional MR blend overlay when an MR series is present beside the CT series
- loading and displaying multiple RTSTRUCT ROIs such as GTV, CTV, and PTV at the same time
- generating an artificial uncertainty layer from the selected contour
- editing a cloned ROI with an add/erase brush on the current slice
- exporting the edited contour as a new RTSTRUCT file

## Suggested libraries for this app

- `PySide6` for the desktop UI
- `pydicom` for DICOM metadata and file IO
- `rt-utils` for RTSTRUCT mask extraction
- `numpy` and `scipy` for mask logic, geometry, and uncertainty generation
- `scikit-image` for contour polygon extraction during RTSTRUCT export

If you later want a more advanced MPR/3D rendering stack, the next libraries worth evaluating are `VTK`, `SimpleITK`, or a web front end with `cornerstone3D`.

## Migration note

### What moved/refactored

1. **Centralized DICOM utilities**
   - Slice discovery/sorting and metadata extraction were consolidated into `dicom_utils.py`.
2. **Consistent slice ordering in extraction**
   - `roi_extraction._load_geometry()` and `_load_ct_volume()` now share the same orientation-based sorting strategy using ImageOrientationPatient + ImagePositionPatient projections.
3. **Validation improvements**
   - Added explicit validation for geometry consistency, non-positive spacing, and warnings for nonuniform slice spacing.
4. **GUI-facing loading API**
   - Added `load_study()` to build a single structured object for desktop usage.
5. **Export wrapper**
   - Added `ExportService` to expose a cleaner API for app-driven ROI export while reusing existing DICOM export logic.

### Why

- Reduce duplicated DICOM logic and reduce subtle geometry mismatches.
- Provide typed, auditable data contracts for GUI and editing flows.
- Enable incremental UI/editor development without rewriting proven domain algorithms.
