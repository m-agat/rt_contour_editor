"""
dicom_export.py
===============
Export uncertainty maps to DICOM formats compatible with Varian Eclipse,
RayStation, and Elekta Monaco.

Two outputs per ROI:
    1. RT Dose  (SOPClassUID 1.2.840.10008.5.1.4.1.1.481.2)
       – continuous uncertainty heatmap (float32 → scaled uint32 grid)
       – All three TPS can import RT Dose volumes as overlay / evaluation dose.

    2. RT Structure Set  (SOPClassUID 1.2.840.10008.5.1.4.1.1.481.3)
       – binary uncertain contour segments as a named ROI ("Uncertain_Boundary")
       – Allows the radiotherapist to see the flagged segments alongside the
         planning structures in the TPS contouring workspace.

Geometry is taken directly from the original CT series so that the exported
DICOM files share the same Frame of Reference UID as the planning CT and any
existing RTSTRUCT / RT Plan files in the same study.
"""

from __future__ import annotations

import datetime
import hashlib
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    UID,
)

# ---------------------------------------------------------------------------
# SOP Class UIDs (well-known, do not change)
# ---------------------------------------------------------------------------
RT_DOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
RT_STRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"

# ---------------------------------------------------------------------------
# UID helpers
# ---------------------------------------------------------------------------

def _uid_from_str(s: str) -> str:
    """Deterministic UID derived from an arbitrary string (for reproducibility)."""
    digest = hashlib.md5(s.encode()).hexdigest()
    # Convert to a numeric string and cap at 64 chars (DICOM limit)
    num = str(int(digest, 16))
    prefix = "2.25."
    return (prefix + num)[:64]


def _new_uid() -> str:
    return generate_uid()


def _now_str() -> tuple[str, str]:
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S.%f")[:13]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _image_position_patient(origin_xyz: np.ndarray, z_index: int,
                              spacing_xyz: np.ndarray,
                              direction: np.ndarray | None) -> list[str]:
    """
    Return ImagePositionPatient (x,y,z in mm) for a given Z slice index.
    origin_xyz : (x0, y0, z0) of the first voxel.
    spacing_xyz: (dx, dy, dz).
    direction  : 9-element row-major direction cosines (SimpleITK convention).
    """
    if direction is None or len(direction) != 9:
        # Assume axial, LPS
        z_mm = float(origin_xyz[2]) + z_index * float(spacing_xyz[2])
        return [f"{origin_xyz[0]:.6f}", f"{origin_xyz[1]:.6f}", f"{z_mm:.6f}"]
    d = np.array(direction, dtype=float).reshape(3, 3)
    pos = np.array(origin_xyz, dtype=float) + z_index * float(spacing_xyz[2]) * d[2]
    return [f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"]


def _image_orientation_patient(direction: np.ndarray | None) -> list[str]:
    """Row and column cosines as required by ImageOrientationPatient."""
    if direction is None or len(direction) != 9:
        return ["1", "0", "0", "0", "1", "0"]  # standard axial LPS
    d = np.array(direction, dtype=float).reshape(3, 3)
    row = d[0]
    col = d[1]
    return [f"{v:.6f}" for v in list(row) + list(col)]


# ---------------------------------------------------------------------------
# Minimal CT ReferencedSOPSequence builder
# ---------------------------------------------------------------------------

def _build_ct_ref_sequence(ct_sop_uids: list[str]) -> DicomSequence:
    items = []
    for uid_str in ct_sop_uids:
        item = Dataset()
        item.ReferencedSOPClassUID = CT_SOP_CLASS
        item.ReferencedSOPInstanceUID = uid_str
        items.append(item)
    return DicomSequence(items)


# ---------------------------------------------------------------------------
# RT Dose export (heatmap)
# ---------------------------------------------------------------------------

def export_rtdose_uncertainty(
    out_path: Path,
    pmap_3d: np.ndarray,            # float32 [0,1], shape (Z, Y, X)
    origin_xyz: Sequence[float],    # (x0, y0, z0) mm – first voxel
    spacing_xyz: Sequence[float],   # (dx, dy, dz) mm
    direction: np.ndarray | None,   # 9-element or None
    frame_of_ref_uid: str,
    study_instance_uid: str,
    series_instance_uid: str | None = None,
    patient_id: str = "UNKNOWN",
    patient_name: str = "UNKNOWN",
    roi_name: str = "ROI",
    ct_sop_uids: list[str] | None = None,
    dose_units: str = "RELATIVE",   # "RELATIVE" → dimensionless [0,1]
) -> None:
    """
    Write a DICOM RT Dose file encoding the continuous uncertainty heatmap.

    The float [0,1] values are scaled to uint32 with DoseGridScaling so that
    original_value = pixel_value * DoseGridScaling.  This preserves full
    floating-point precision within the 32-bit integer range.

    Parameters
    ----------
    out_path        : Destination .dcm file path.
    pmap_3d         : 3-D uncertainty array, float32, shape (Z, Y, X), values in [0,1].
    origin_xyz      : (x, y, z) of the first voxel in mm (LPS / patient coordinates).
    spacing_xyz     : (dx, dy, dz) voxel size in mm.
    direction       : Optional 9-element row-major direction cosine array.
    frame_of_ref_uid: FrameOfReferenceUID matching the planning CT.
    study_instance_uid : StudyInstanceUID matching the planning CT study.
    series_instance_uid: If None, a new UID is generated.
    patient_id      : PatientID for DICOM header.
    patient_name    : PatientName for DICOM header.
    roi_name        : Used in SeriesDescription for human-readable labelling.
    ct_sop_uids     : SOPInstanceUIDs of CT slices, used for ReferencedInstanceSequence.
    dose_units      : "RELATIVE" (default) or "GY" / "CGY" – set to RELATIVE for
                      dimensionless maps; TPS will display it as a ratio.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pmap = np.asarray(pmap_3d, dtype=np.float32)
    n_z, n_y, n_x = pmap.shape

    # Scale float [0,1] → uint32
    MAX_UINT32 = 2**32 - 1
    pmax = float(pmap.max())
    if pmax < 1e-9:
        pmax = 1.0
    grid_scaling = pmax / MAX_UINT32   # DoseGridScaling
    pixel_array = np.round(pmap / grid_scaling).astype(np.uint32)  # (Z, Y, X)

    date_str, time_str = _now_str()
    sop_instance_uid = _new_uid()
    series_uid = series_instance_uid or _new_uid()

    # ---- File meta ----
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(RT_DOSE_SOP_CLASS)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # ---- Patient / study ----
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.StudyDescription = "Uncertainty Maps"
    ds.ReferringPhysicianName = ""
    ds.AccessionNumber = ""

    # ---- Series ----
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = "900"
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.SeriesDescription = f"UncHeatmap_{roi_name}"
    ds.Modality = "RTDOSE"
    ds.Manufacturer = ""

    # ---- SOP common ----
    ds.SOPClassUID = RT_DOSE_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str
    ds.SpecificCharacterSet = "ISO_IR 192"

    # ---- Frame of reference ----
    ds.FrameOfReferenceUID = frame_of_ref_uid
    ds.PositionReferenceIndicator = ""

    # ---- Image geometry ----
    origin = list(origin_xyz)
    spacing = list(spacing_xyz)

    ds.Rows = n_y
    ds.Columns = n_x
    ds.NumberOfFrames = n_z
    ds.PixelSpacing = [f"{spacing[1]:.6f}", f"{spacing[0]:.6f}"]  # row, col spacing (dy, dx)
    ds.SliceThickness = f"{spacing[2]:.6f}"
    ds.ImageOrientationPatient = _image_orientation_patient(direction)
    ds.ImagePositionPatient = _image_position_patient(
        np.array(origin), 0, np.array(spacing), direction
    )
    ds.GridFrameOffsetVector = [
        f"{i * spacing[2]:.6f}" for i in range(n_z)
    ]

    # ---- Dose-specific ----
    ds.DoseUnits = dose_units
    ds.DoseType = "PHYSICAL"          # most TPSs accept; alternatives: EFFECTIVE, ERROR
    ds.DoseSummationType = "PLAN"     # required tag; value is nominal here
    ds.DoseGridScaling = f"{grid_scaling:.10e}"
    ds.TissueHeterogeneityCorrection = []

    # ---- Pixel data ----
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0        # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # pixel_array must be (Z, Y, X) → flatten in C order
    ds.PixelData = pixel_array.tobytes()

    # Optional: reference the planning CT slices
    if ct_sop_uids:
        ref_study = Dataset()
        ref_study.ReferencedSOPClassUID = CT_SOP_CLASS
        ref_study.ReferencedSOPInstanceUID = study_instance_uid
        ref_study.ReferencedInstanceSequence = _build_ct_ref_sequence(ct_sop_uids)
        ds.ReferencedStudySequence = DicomSequence([ref_study])

    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)
    print(f"[RT Dose] Saved heatmap  → {out_path}")


# ---------------------------------------------------------------------------
# RT Structure Set export (binary uncertain contour segments)
# ---------------------------------------------------------------------------

def _mask_to_contour_sequences(
    binary_mask_3d: np.ndarray,    # bool, (Z, Y, X)
    origin_xyz: Sequence[float],   # (x0, y0, z0)
    spacing_xyz: Sequence[float],  # (dx, dy, dz)
    direction: np.ndarray | None,
    min_points: int = 3,
) -> tuple[list[Dataset], list[Dataset]]:
    """
    Convert a binary 3-D mask to DICOM RT contour sequences (one contour per
    connected slice outline).  Returns (ROIContourSequence_items, per-slice datasets).

    Implementation: for each Z slice we use a simple pixel-boundary walk.
    We delegate the polygon extraction to skimage.measure.find_contours which
    returns sub-pixel coordinates, then we snap them to the nearest voxel centre
    and project to patient coordinates.
    """
    try:
        from skimage.measure import find_contours as ski_find_contours
        _have_ski = True
    except ImportError:
        _have_ski = False

    origin = np.array(origin_xyz, dtype=float)
    spacing = np.array(spacing_xyz, dtype=float)

    contour_datasets: list[Dataset] = []

    for z_idx in range(binary_mask_3d.shape[0]):
        slice_mask = binary_mask_3d[z_idx].astype(bool)
        if not slice_mask.any():
            continue

        if _have_ski:
            # find_contours returns (row, col) = (y, x) in pixel coords
            polys = ski_find_contours(slice_mask.astype(float), level=0.5)
        else:
            # Minimal fallback: bounding box corners of each connected component
            from scipy.ndimage import label as nd_label
            labeled, n = nd_label(slice_mask)
            polys = []
            for lbl in range(1, n + 1):
                ys, xs = np.where(labeled == lbl)
                # Simple convex rectangle outline
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                polys.append(np.array([
                    [y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0]
                ], dtype=float))

        for poly in polys:
            if len(poly) < min_points:
                continue
            # poly shape: (N, 2) as (row=y_px, col=x_px)
            # Project to patient coords (LPS)
            coords_3d = []
            for row_px, col_px in poly:
                # Voxel index → patient xyz (simplified: axial LPS)
                if direction is not None and len(direction) == 9:
                    d = np.array(direction).reshape(3, 3)
                    pt = (origin
                          + col_px  * spacing[0] * d[0]
                          + row_px  * spacing[1] * d[1]
                          + z_idx   * spacing[2] * d[2])
                else:
                    pt = np.array([
                        origin[0] + col_px * spacing[0],
                        origin[1] + row_px * spacing[1],
                        origin[2] + z_idx  * spacing[2],
                    ])
                coords_3d.extend([f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])

            if len(coords_3d) < min_points * 3:
                continue

            cds = Dataset()
            cds.ContourGeometricType = "CLOSED_PLANAR"
            cds.NumberOfContourPoints = len(poly)
            cds.ContourData = coords_3d
            contour_datasets.append(cds)

    return contour_datasets


def export_rtstruct_uncertainty(
    out_path: Path,
    binary_mask_3d: np.ndarray,      # bool / uint8, (Z, Y, X) — uncertain boundary voxels
    origin_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
    direction: np.ndarray | None,
    frame_of_ref_uid: str,
    study_instance_uid: str,
    series_instance_uid: str | None = None,
    patient_id: str = "UNKNOWN",
    patient_name: str = "UNKNOWN",
    roi_name: str = "ROI",
    referenced_rtstruct_sop_uid: str | None = None,  # original planning RTSTRUCT SOP UID if known
    ct_series_uid: str | None = None,
    ct_sop_uids: list[str] | None = None,
    struct_color: tuple[int, int, int] = (255, 165, 0),   # orange to match BIN_COLOR
) -> None:
    """
    Write a DICOM RT Structure Set file containing the binary uncertain contour
    segments as an ROI named ``Uncertain_Boundary_<roi_name>``.

    The TPS operator can load this RTSTRUCT alongside the planning RTSTRUCT and
    compare the flagged boundary regions directly in the contouring workspace.

    Parameters
    ----------
    out_path              : Destination .dcm file.
    binary_mask_3d        : Boolean 3-D mask of uncertain boundary voxels (Z, Y, X).
    origin_xyz            : (x, y, z) of first voxel in mm.
    spacing_xyz           : (dx, dy, dz) in mm.
    direction             : 9-element direction cosines or None (→ axial LPS).
    frame_of_ref_uid      : Must match planning CT FrameOfReferenceUID.
    study_instance_uid    : Must match planning CT StudyInstanceUID.
    struct_color          : RGB display colour for the TPS (default: orange).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    date_str, time_str = _now_str()
    sop_instance_uid = _new_uid()
    series_uid = series_instance_uid or _new_uid()

    mask = np.asarray(binary_mask_3d, dtype=bool)
    struct_label = f"Uncertain_Boundary_{roi_name}"

    # ---- File meta ----
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(RT_STRUCT_SOP_CLASS)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # ---- Patient / study ----
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.StudyDescription = "Uncertainty Maps"
    ds.ReferringPhysicianName = ""
    ds.AccessionNumber = ""

    # ---- Series ----
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = "901"
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.SeriesDescription = f"UncBinary_{roi_name}"
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = ""

    # ---- SOP common ----
    ds.SOPClassUID = RT_STRUCT_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str
    ds.SpecificCharacterSet = "ISO_IR 192"

    # ---- Frame of reference ----
    frame_ref_ds = Dataset()
    frame_ref_ds.FrameOfReferenceUID = frame_of_ref_uid
    frame_ref_ds.PositionReferenceIndicator = ""
    ds.ReferencedFrameOfReferenceSequence = DicomSequence([frame_ref_ds])

    # ---- Referenced series (planning CT) ----
    if ct_series_uid or ct_sop_uids:
        ref_series_ds = Dataset()
        ref_series_ds.SeriesInstanceUID = ct_series_uid or _new_uid()
        if ct_sop_uids:
            ref_series_ds.ReferencedInstanceSequence = _build_ct_ref_sequence(ct_sop_uids)
        ref_study_ds = Dataset()
        ref_study_ds.StudyInstanceUID = study_instance_uid
        ref_study_ds.RTReferencedSeriesSequence = DicomSequence([ref_series_ds])
        frame_ref_ds.RTReferencedStudySequence = DicomSequence([ref_study_ds])

    # ---- Structure Set ----
    ds.StructureSetLabel = "UncertaintyMask"
    ds.StructureSetName = "UncertaintyMask"
    ds.StructureSetDate = date_str
    ds.StructureSetTime = time_str

    # ---- ROI definitions ----
    roi_number = 1
    roi_ds = Dataset()
    roi_ds.ROINumber = roi_number
    roi_ds.ReferencedFrameOfReferenceUID = frame_of_ref_uid
    roi_ds.ROIName = struct_label
    roi_ds.ROIDescription = (
        f"Top {int(0.25*100)}% most-uncertain AI boundary voxels for {roi_name}. "
        "Generated automatically – not a clinical contour."
    )
    roi_ds.ROIGenerationAlgorithm = "AUTOMATIC"
    ds.StructureSetROISequence = DicomSequence([roi_ds])

    # ---- ROI observations ----
    obs_ds = Dataset()
    obs_ds.ObservationNumber = roi_number
    obs_ds.ReferencedROINumber = roi_number
    obs_ds.ROIObservationLabel = struct_label
    obs_ds.RTROIInterpretedType = "CONTROL"   # neutral type; not a clinical target
    obs_ds.ROIInterpreter = ""
    ds.RTROIObservationsSequence = DicomSequence([obs_ds])

    # ---- Contour data ----
    contour_data = _mask_to_contour_sequences(mask, origin_xyz, spacing_xyz, direction)

    roi_contour_ds = Dataset()
    roi_contour_ds.ReferencedROINumber = roi_number
    roi_contour_ds.ROIDisplayColor = list(struct_color)
    roi_contour_ds.ContourSequence = DicomSequence(contour_data) if contour_data else DicomSequence([])
    ds.ROIContourSequence = DicomSequence([roi_contour_ds])

    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)
    print(f"[RT Struct] Saved binary → {out_path}  ({len(contour_data)} contour slices)")


# ---------------------------------------------------------------------------
# Convenience wrapper — call from main.py
# ---------------------------------------------------------------------------

def export_uncertainty_to_dicom(
    out_dir: Path,
    roi_name: str,
    pmap: np.ndarray,                    # (Z,Y,X) float32 [0,1] – heatmap
    contour_bin: np.ndarray,             # (Z,Y,X) bool/uint8   – binary uncertain boundary
    origin: np.ndarray | Sequence[float],   # (x,y,z) mm
    spacing: np.ndarray | Sequence[float],  # (dx,dy,dz) mm
    direction: np.ndarray | None,
    frame_of_ref_uid: str,
    study_instance_uid: str,
    patient_id: str = "UNKNOWN",
    patient_name: str = "UNKNOWN",
    ct_series_uid: str | None = None,
    ct_sop_uids: list[str] | None = None,
) -> None:
    """
    High-level entry point called from main.py after compute_pmap().

    Writes two files into out_dir:
        {roi_name}_unc_heatmap.dcm   – RT Dose (continuous heatmap, DoseGridScaling in [0,1])
        {roi_name}_unc_binary.dcm    – RT Structure Set (binary uncertain boundary ROI)

    Both files carry the same FrameOfReferenceUID and StudyInstanceUID as the
    planning CT, enabling direct import into the TPS without manual registration.
    """
    out_dir = Path(out_dir)

    export_rtdose_uncertainty(
        out_path=out_dir / f"{roi_name}_unc_heatmap.dcm",
        pmap_3d=pmap,
        origin_xyz=origin,
        spacing_xyz=spacing,
        direction=direction,
        frame_of_ref_uid=frame_of_ref_uid,
        study_instance_uid=study_instance_uid,
        patient_id=patient_id,
        patient_name=patient_name,
        roi_name=roi_name,
        ct_sop_uids=ct_sop_uids,
    )

    export_rtstruct_uncertainty(
        out_path=out_dir / f"{roi_name}_unc_binary.dcm",
        binary_mask_3d=contour_bin.astype(bool),
        origin_xyz=origin,
        spacing_xyz=spacing,
        direction=direction,
        frame_of_ref_uid=frame_of_ref_uid,
        study_instance_uid=study_instance_uid,
        patient_id=patient_id,
        patient_name=patient_name,
        roi_name=roi_name,
        ct_series_uid=ct_series_uid,
        ct_sop_uids=ct_sop_uids,
    )