from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rt_utils import RTStructBuilder

from dicom_utils import (
    discover_ct_slices,
    discover_modality_slices,
    discover_series_by_modality,
    extract_series_metadata,
    read_dicom_safe,
    sort_ct_slices,
)
from models import DicomStudyMetadata, ImageSeries, LoadedStudy
from roi_extraction import VolumeGeometry, _load_ct_volume, _load_geometry

LOGGER = logging.getLogger(__name__)


def _load_volume_for_modality(folder: Path, modality: str) -> np.ndarray:
    slices = sort_ct_slices(discover_modality_slices(folder, modality, stop_before_pixels=False))
    if not slices:
        raise RuntimeError(f"No {modality} slices found in {folder}")

    planes = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        planes.append((arr * slope + intercept).astype(np.float32))
    return np.stack(planes, axis=0)


def _load_geometry_for_modality(folder: Path, modality: str) -> VolumeGeometry:
    if modality.upper() == "CT":
        return _load_geometry(folder)

    slices = sort_ct_slices(discover_modality_slices(folder, modality, stop_before_pixels=True))
    if not slices:
        raise RuntimeError(f"No {modality} slices found in {folder}")

    ds0 = slices[0]
    orientation = np.asarray(ds0.ImageOrientationPatient, dtype=float)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    slice_cos = np.cross(row_cos, col_cos)
    slice_cos = slice_cos / np.linalg.norm(slice_cos)

    origin = np.asarray(ds0.ImagePositionPatient, dtype=float)
    dy, dx = [float(x) for x in ds0.PixelSpacing]
    if len(slices) > 1:
        projections = [float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), slice_cos)) for ds in slices]
        dz = float(np.median(np.abs(np.diff(projections))))
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0))

    spacing = np.array([dx, dy, dz], dtype=float)
    direction = np.stack([col_cos, row_cos, slice_cos], axis=1)
    return VolumeGeometry(origin=origin, spacing=spacing, direction=direction)


def _discover_mr_series(ct_folder: Path) -> dict[str, ImageSeries]:
    visit_dir = ct_folder.parent
    discovered = discover_series_by_modality(visit_dir, "MR")
    results: dict[str, ImageSeries] = {}
    for series_uid, files in discovered.items():
        folder = Path(files[0]).parent
        try:
            geometry = _load_geometry_for_modality(folder, "MR")
            volume = _load_volume_for_modality(folder, "MR")
        except Exception as exc:
            LOGGER.warning("Skipping MR series %s in %s: %s", series_uid, folder, exc)
            continue

        series_name = folder.name
        results[series_name] = ImageSeries(
            name=series_name,
            modality="MR",
            series_instance_uid=series_uid,
            volume=volume,
            geometry=geometry,
        )
    return results


def load_study(ct_folder: Path, rtstruct_path: Path, *, eager_roi_masks: bool = True) -> LoadedStudy:
    """Load CT geometry/volume and RTSTRUCT ROIs into a coherent app model."""
    ct_folder = Path(ct_folder)
    rtstruct_path = Path(rtstruct_path)

    ct_slices = sort_ct_slices(discover_ct_slices(ct_folder, stop_before_pixels=True))
    if not ct_slices:
        raise RuntimeError(f"No CT slices found in {ct_folder}")
    ct_meta = extract_series_metadata(ct_slices)

    rt_ds = read_dicom_safe(rtstruct_path, stop_before_pixels=True)
    if rt_ds is None:
        raise RuntimeError(f"Failed to read RTSTRUCT file: {rtstruct_path}")

    rt = RTStructBuilder.create_from(
        dicom_series_path=str(ct_folder),
        rt_struct_path=str(rtstruct_path),
    )
    roi_names = sorted(rt.get_roi_names())
    roi_masks: dict[str, np.ndarray] = {}
    if eager_roi_masks:
        for roi_name in roi_names:
            mask_yxz = rt.get_roi_mask_by_name(roi_name)
            roi_masks[roi_name] = np.transpose(mask_yxz, (2, 0, 1)).astype(bool)

    geometry = _load_geometry(ct_folder)
    ct_volume = _load_ct_volume(ct_folder)
    for roi_name, roi_mask in roi_masks.items():
        if roi_mask.shape != ct_volume.shape:
            raise RuntimeError(
                f"ROI '{roi_name}' shape {roi_mask.shape} does not match CT volume {ct_volume.shape}"
            )

    metadata = DicomStudyMetadata(
        patient_id=ct_meta.patient_id,
        patient_name=ct_meta.patient_name,
        study_instance_uid=ct_meta.study_instance_uid,
        ct_series_instance_uid=ct_meta.series_instance_uid,
        rtstruct_sop_instance_uid=str(getattr(rt_ds, "SOPInstanceUID", "")),
        frame_of_reference_uid=ct_meta.frame_of_reference_uid,
        ct_sop_instance_uids=ct_meta.sop_instance_uids,
    )
    LOGGER.info("Loaded study with %d CT slices and %d ROIs", ct_volume.shape[0], len(roi_names))

    image_series: dict[str, ImageSeries] = {
        "CT": ImageSeries(
            name="CT",
            modality="CT",
            series_instance_uid=ct_meta.series_instance_uid,
            volume=ct_volume.astype(np.float32),
            geometry=geometry,
        )
    }
    image_series.update(_discover_mr_series(ct_folder))

    return LoadedStudy(
        ct_volume=ct_volume,
        geometry=geometry,
        roi_masks=roi_masks,
        roi_names=roi_names,
        image_series=image_series,
        ct_folder=ct_folder,
        rtstruct_path=rtstruct_path,
        metadata=metadata,
    )
