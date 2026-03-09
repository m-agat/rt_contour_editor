# rt_contour_editor

## Desktop-app foundation architecture

This repository now includes a modular foundation for a PySide6 glioma contour viewer/editor while preserving existing extraction/export logic.

- `dicom_utils.py`: shared DICOM IO, CT discovery, orientation-aware slice sorting, spacing validation, and metadata extraction.
- `roi_extraction.py`: ROI extraction built on consistent geometry/slice ordering utilities.
- `models.py`: dataclass app models (`DicomStudyMetadata`, `RoiLayer`, `LoadedStudy`).
- `study_loader.py`: high-level `load_study(ct_folder, rtstruct_path)` API returning one coherent in-memory study.
- `viewer_state.py` and `editor_state.py`: app/view/edit state models for upcoming brush editing.
- `export_service.py`: app service wrapper to export edited ROI masks using existing `dicom_export.py` logic.
- `gui/`: minimal PySide6 MVP (`main_window.py`, `slice_canvas.py`) for loading CT+RTSTRUCT, scrolling slices, and ROI overlay viewing.
- `app.py`: desktop entrypoint.

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
