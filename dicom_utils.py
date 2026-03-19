from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DicomSeriesMetadata:
    """Metadata extracted from a CT series for downstream model loading/export."""

    patient_id: str
    patient_name: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    sop_instance_uids: list[str]


def read_dicom_safe(path: Path, *, stop_before_pixels: bool = True) -> Dataset | None:
    """Read a DICOM file and return ``None`` for unreadable/non-DICOM inputs."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.debug("Skipping unreadable DICOM %s: %s", path, exc)
        return None


def discover_ct_slices(ct_folder: Path, *, stop_before_pixels: bool = False) -> list[Dataset]:
    """Load all CT datasets from ``ct_folder`` recursively."""
    slices: list[Dataset] = []
    for file_path in sorted(ct_folder.rglob("*")):
        ds = read_dicom_safe(file_path, stop_before_pixels=stop_before_pixels)
        if ds is None:
            continue
        if getattr(ds, "Modality", "").upper() != "CT":
            continue
        slices.append(ds)
    return slices


def discover_modality_slices(
    folder: Path,
    modality: str,
    *,
    stop_before_pixels: bool = False,
) -> list[Dataset]:
    """Load all DICOM datasets with the requested modality from ``folder`` recursively."""
    wanted = modality.upper()
    slices: list[Dataset] = []
    for file_path in sorted(folder.rglob("*")):
        ds = read_dicom_safe(file_path, stop_before_pixels=stop_before_pixels)
        if ds is None:
            continue
        if getattr(ds, "Modality", "").upper() != wanted:
            continue
        slices.append(ds)
    return slices


def discover_series_by_modality(visit_dir: Path, modality: str) -> dict[str, list[Path]]:
    """Index all series for ``modality`` under ``visit_dir`` as ``{SeriesInstanceUID: files}``."""
    wanted = modality.upper()
    indexed: dict[str, list[Path]] = {}
    for file_path in sorted(Path(visit_dir).rglob("*")):
        ds = read_dicom_safe(file_path, stop_before_pixels=True)
        if ds is None:
            continue
        if getattr(ds, "Modality", "").upper() != wanted:
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        if not series_uid:
            continue
        indexed.setdefault(series_uid, []).append(file_path)
    return indexed


def _require_orientation(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(ds, "ImageOrientationPatient"):
        raise ValueError("CT slice missing ImageOrientationPatient")
    if not hasattr(ds, "ImagePositionPatient"):
        raise ValueError("CT slice missing ImagePositionPatient")

    orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
    if orient.size != 6:
        raise ValueError("ImageOrientationPatient must have 6 values")

    row_cos = orient[:3]
    col_cos = orient[3:]
    slice_cos = np.cross(row_cos, col_cos)
    norm = np.linalg.norm(slice_cos)
    if norm < 1e-6:
        raise ValueError("Invalid orientation cosines: row/column vectors are colinear")

    return row_cos, col_cos, slice_cos / norm


def slice_position_projection(ds: Dataset, slice_normal: np.ndarray) -> float:
    """Projection of ImagePositionPatient onto slice normal."""
    ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
    return float(np.dot(ipp, slice_normal))


def sort_ct_slices(slices: list[Dataset]) -> list[Dataset]:
    """Return CT slices sorted by projection along the orientation-derived normal."""
    if not slices:
        raise ValueError("No CT slices available for sorting")
    _, _, slice_normal = _require_orientation(slices[0])
    return sorted(slices, key=lambda ds: slice_position_projection(ds, slice_normal))


def validate_uniform_slice_spacing(slices: list[Dataset], *, tolerance_mm: float = 1e-3) -> float:
    """Validate slice spacing is near-uniform and return nominal dz in mm."""
    if not slices:
        raise ValueError("No CT slices available for spacing validation")

    _, _, normal = _require_orientation(slices[0])
    projections = [slice_position_projection(ds, normal) for ds in sort_ct_slices(slices)]
    if len(projections) <= 1:
        return float(getattr(slices[0], "SliceThickness", 1.0))

    diffs = np.diff(projections)
    dz = float(np.median(np.abs(diffs)))
    if np.any(np.abs(np.abs(diffs) - dz) > tolerance_mm):
        LOGGER.warning(
            "Nonuniform CT slice spacing detected: diffs=%s, nominal=%.6f mm",
            np.array2string(diffs, precision=6),
            dz,
        )
    return dz


def extract_series_metadata(ct_slices: list[Dataset]) -> DicomSeriesMetadata:
    """Extract identifying metadata from a sorted CT series."""
    if not ct_slices:
        raise ValueError("Cannot extract metadata from an empty CT series")

    first = ct_slices[0]
    return DicomSeriesMetadata(
        patient_id=str(getattr(first, "PatientID", "UNKNOWN")),
        patient_name=str(getattr(first, "PatientName", "UNKNOWN")),
        study_instance_uid=str(getattr(first, "StudyInstanceUID", "")),
        series_instance_uid=str(getattr(first, "SeriesInstanceUID", "")),
        frame_of_reference_uid=str(getattr(first, "FrameOfReferenceUID", "")),
        sop_instance_uids=[str(getattr(ds, "SOPInstanceUID", "")) for ds in ct_slices],
    )


def as_float_sequence(value: Any, *, expected_len: int, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size != expected_len:
        raise ValueError(f"{field_name} must have {expected_len} values")
    return arr
