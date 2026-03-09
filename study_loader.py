from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rt_utils import RTStructBuilder

from dicom_utils import discover_ct_slices, extract_series_metadata, read_dicom_safe, sort_ct_slices
from models import DicomStudyMetadata, LoadedStudy
from roi_extraction import _load_ct_volume, _load_geometry

LOGGER = logging.getLogger(__name__)


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

    return LoadedStudy(
        ct_volume=ct_volume,
        geometry=geometry,
        roi_masks=roi_masks,
        roi_names=roi_names,
        ct_folder=ct_folder,
        rtstruct_path=rtstruct_path,
        metadata=metadata,
    )
