from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from rt_utils import RTStructBuilder

from dicom_utils import (
    as_float_sequence,
    discover_ct_slices,
    sort_ct_slices,
    validate_uniform_slice_spacing,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class VolumeGeometry:
    """
    Physical geometry of a 3D volume in patient (LPS) coordinates.

    origin      : (3,) float  — position of voxel [0,0,0] in mm
    spacing     : (3,) float  — voxel size (x, y, z) in mm
    direction   : (3,3) float — row/col/slice direction cosines (rows = axes)
    """

    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray

    def voxel_to_mm(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=float)
        return self.origin + self.direction @ (self.spacing * ijk)

    def mm_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return (np.linalg.inv(self.direction) @ (xyz - self.origin)) / self.spacing


@dataclass
class RoiExtractionResult:
    """CT volume, extracted mask, geometry, and selected ROI name."""

    ct_volume: np.ndarray
    roi_mask: np.ndarray
    geometry: VolumeGeometry
    roi_name: str


def _load_geometry(ct_folder: Path) -> VolumeGeometry:
    slices = sort_ct_slices(discover_ct_slices(ct_folder, stop_before_pixels=True))
    if not slices:
        raise RuntimeError(f"No CT slices found in {ct_folder}")

    ds0 = slices[0]
    orientation = as_float_sequence(ds0.ImageOrientationPatient, expected_len=6, field_name="ImageOrientationPatient")
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    slice_cos = np.cross(row_cos, col_cos)
    slice_cos = slice_cos / np.linalg.norm(slice_cos)

    origin = as_float_sequence(ds0.ImagePositionPatient, expected_len=3, field_name="ImagePositionPatient")
    dy, dx = [float(x) for x in ds0.PixelSpacing]
    dz = validate_uniform_slice_spacing(slices)

    spacing = np.array([dx, dy, dz], dtype=float)
    direction = np.stack([col_cos, row_cos, slice_cos], axis=1)
    return VolumeGeometry(origin=origin, spacing=spacing, direction=direction)


def _load_ct_volume(ct_folder: Path) -> np.ndarray:
    """Load CT slices into a ``(Z, Y, X)`` HU volume using orientation sort order."""
    slices = sort_ct_slices(discover_ct_slices(ct_folder, stop_before_pixels=False))
    if not slices:
        raise RuntimeError(f"No CT slices found in {ct_folder}")

    planes = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        planes.append((arr * slope + intercept).astype(np.int16))

    return np.stack(planes, axis=0)


def _validate_alignment(ct_volume: np.ndarray, roi_mask: np.ndarray, geometry: VolumeGeometry) -> None:
    if ct_volume.shape != roi_mask.shape:
        raise RuntimeError(
            f"Shape mismatch: CT {ct_volume.shape} vs mask {roi_mask.shape}. "
            "Check that the RTSTRUCT references this CT series."
        )
    if np.any(geometry.spacing <= 0):
        raise RuntimeError(f"Invalid non-positive spacing detected: {geometry.spacing}")
    det = float(np.linalg.det(geometry.direction))
    if abs(det) < 1e-6:
        raise RuntimeError("Invalid geometry direction matrix (singular)")
    LOGGER.debug("Validated ROI alignment for volume shape %s", ct_volume.shape)


def extract_roi(ct_folder: Path, rtstruct_path: Path, roi_name: str) -> RoiExtractionResult:
    """Extract one ROI mask and CT volume from matched CT/RTSTRUCT inputs."""
    ct_folder = Path(ct_folder)
    rtstruct_path = Path(rtstruct_path)

    rt = RTStructBuilder.create_from(
        dicom_series_path=str(ct_folder),
        rt_struct_path=str(rtstruct_path),
    )

    available = rt.get_roi_names()
    if roi_name not in available:
        raise ValueError(f"ROI '{roi_name}' not found. Available: {available}")

    mask_yxz = rt.get_roi_mask_by_name(roi_name)
    roi_mask = np.transpose(mask_yxz, (2, 0, 1)).astype(bool)

    ct_volume = _load_ct_volume(ct_folder)
    geometry = _load_geometry(ct_folder)
    _validate_alignment(ct_volume, roi_mask, geometry)

    return RoiExtractionResult(
        ct_volume=ct_volume,
        roi_mask=roi_mask,
        geometry=geometry,
        roi_name=roi_name,
    )
