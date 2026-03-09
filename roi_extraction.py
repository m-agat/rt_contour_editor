from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from rt_utils import RTStructBuilder


@dataclass
class VolumeGeometry:
    """
    Physical geometry of a 3D volume in patient (LPS) coordinates.

    origin      : (3,) float  — position of voxel [0,0,0] in mm
    spacing     : (3,) float  — voxel size (x, y, z) in mm
    direction   : (3,3) float — row/col/slice direction cosines (rows = axes)

    Use `voxel_to_mm(ijk)` and `mm_to_voxel(xyz)` for coordinate conversion.
    """
    origin: np.ndarray    # shape (3,)
    spacing: np.ndarray   # shape (3,) — (dx, dy, dz) in mm
    direction: np.ndarray # shape (3,3)

    def voxel_to_mm(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=float)
        return self.origin + self.direction @ (self.spacing * ijk)

    def mm_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return (np.linalg.inv(self.direction) @ (xyz - self.origin)) / self.spacing


@dataclass
class RoiExtractionResult:
    """
    Everything you need for SDF-based contour perturbation.

    ct_volume   : (Z, Y, X) int16  — HU values
    roi_mask    : (Z, Y, X) bool   — True inside the ROI
    geometry    : VolumeGeometry   — voxel <-> mm mapping
    roi_name    : str              — name as stored in RTSTRUCT
    """
    ct_volume: np.ndarray   # (Z, Y, X) int16
    roi_mask: np.ndarray    # (Z, Y, X) bool
    geometry: VolumeGeometry
    roi_name: str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _load_geometry(ct_folder: Path) -> VolumeGeometry:
    slices = []
    for f in ct_folder.rglob("*"):
        ds = _read_dicom(f)
        if ds is None:
            continue
        if getattr(ds, "Modality", "").upper() != "CT":
            continue
        if not (hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelSpacing")):
            continue
        slices.append(ds)

    if not slices:
        raise RuntimeError(f"No CT slices found in {ct_folder}")

    ds0 = slices[0]
    row_cos = np.array(ds0.ImageOrientationPatient[:3], dtype=float)
    col_cos = np.array(ds0.ImageOrientationPatient[3:], dtype=float)
    slice_cos = np.cross(row_cos, col_cos)

    def slice_pos(ds):
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(ipp, slice_cos))

    slices.sort(key=slice_pos)

    ds0 = slices[0]
    ds1 = slices[1] if len(slices) > 1 else slices[0]

    origin = np.array(ds0.ImagePositionPatient, dtype=float)

    # PixelSpacing = [dy, dx]
    dy, dx = [float(x) for x in ds0.PixelSpacing]

    if len(slices) > 1:
        pos0 = np.array(ds0.ImagePositionPatient, dtype=float)
        pos1 = np.array(ds1.ImagePositionPatient, dtype=float)
        dz = float(abs(np.dot(pos1 - pos0, slice_cos)))
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0))

    # spacing for voxel indices (i,j,k) = (x/col, y/row, z/slice)
    spacing = np.array([dx, dy, dz], dtype=float)  # (dx, dy, dz)

    # direction columns correspond to i,j,k axes
    direction = np.stack([col_cos, row_cos, slice_cos], axis=1)  # (3,3)

    return VolumeGeometry(origin=origin, spacing=spacing, direction=direction)

def _load_ct_volume(ct_folder: Path) -> np.ndarray:
    """
    Load CT slices into a (Z, Y, X) int16 array of HU values.
    Applies RescaleSlope / RescaleIntercept if present.
    """
    slices = []
    for f in sorted(ct_folder.rglob("*")):
        ds = _read_dicom(f, stop_before_pixels=False)
        if ds is not None and getattr(ds, "Modality", "").upper() == "CT":
            slices.append(ds)

    slices.sort(key=lambda ds: float(getattr(ds, "SliceLocation",
                                             ds.ImagePositionPatient[2])))

    planes = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        planes.append((arr * slope + intercept).astype(np.int16))

    return np.stack(planes, axis=0)  # (Z, Y, X)


def _read_dicom(path: Path, *, stop_before_pixels: bool = True):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_roi(
    ct_folder: Path,
    rtstruct_path: Path,
    roi_name: str,
) -> RoiExtractionResult:
    """
    Extract a single ROI as a 3D binary mask alongside the CT volume and geometry.

    Args:
        ct_folder     : folder containing the CT DICOM series.
        rtstruct_path : path to the RTSTRUCT DICOM file.
        roi_name      : exact ROI name as returned by RTStructBuilder.get_roi_names().

    Returns:
        RoiExtractionResult with:
          - ct_volume  (Z, Y, X) int16  in HU
          - roi_mask   (Z, Y, X) bool
          - geometry   VolumeGeometry for mm <-> voxel conversion
          - roi_name   str
    """
    ct_folder = Path(ct_folder)
    rtstruct_path = Path(rtstruct_path)

    rt = RTStructBuilder.create_from(
        dicom_series_path=str(ct_folder),
        rt_struct_path=str(rtstruct_path),
    )

    available = rt.get_roi_names()
    if roi_name not in available:
        raise ValueError(
            f"ROI '{roi_name}' not found.\n"
            f"Available: {available}"
        )

    # rt-utils returns (Y, X, Z) — transpose to (Z, Y, X)
    mask_yxz = rt.get_roi_mask_by_name(roi_name)
    roi_mask = np.transpose(mask_yxz, (2, 0, 1)).astype(bool)

    ct_volume = _load_ct_volume(ct_folder)
    geometry = _load_geometry(ct_folder)

    if ct_volume.shape != roi_mask.shape:
        raise RuntimeError(
            f"Shape mismatch: CT {ct_volume.shape} vs mask {roi_mask.shape}. "
            "Check that the RTSTRUCT references this CT series."
        )

    return RoiExtractionResult(
        ct_volume=ct_volume,
        roi_mask=roi_mask,
        geometry=geometry,
        roi_name=roi_name,
    )


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Extract an ROI mask + CT volume from a DICOM RT dataset."
#     )
#     parser.add_argument("--ct-folder",    required=True, help="CT series folder")
#     parser.add_argument("--rtstruct",     required=True, help="RTSTRUCT .dcm file")
#     parser.add_argument("--roi",          required=True, help="ROI name to extract")
#     parser.add_argument("--out",          default=None,  help="Save result as .npz (optional)")
#     args = parser.parse_args()

#     result = extract_roi(
#         ct_folder=Path(args.ct_folder),
#         rtstruct_path=Path(args.rtstruct),
#         roi_name=args.roi,
#     )

#     print(f"ROI name      : {result.roi_name}")
#     print(f"CT shape      : {result.ct_volume.shape}  (Z, Y, X)")
#     print(f"Mask shape    : {result.roi_mask.shape}  (Z, Y, X)")
#     print(f"Voxel count   : {result.roi_mask.sum()}")
#     print(f"Volume (cm³)  : {result.roi_mask.sum() * np.prod(result.geometry.spacing) / 1000:.2f}")
#     print(f"Origin (mm)   : {result.geometry.origin}")
#     print(f"Spacing (mm)  : {result.geometry.spacing}")

#     if args.out:
#         np.savez_compressed(
#             args.out,
#             ct_volume=result.ct_volume,
#             roi_mask=result.roi_mask,
#             origin=result.geometry.origin,
#             spacing=result.geometry.spacing,
#             direction=result.geometry.direction,
#         )
#         print(f"Saved to      : {args.out}")