from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from dicom_export import export_rtstruct_uncertainty
from models import LoadedStudy

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportService:
    """Application-facing export wrapper built on top of ``dicom_export``."""

    out_dir: Path

    def export_edited_roi(self, study: LoadedStudy, roi_name: str, roi_mask: np.ndarray) -> Path:
        """Export ``roi_mask`` as a standalone RTSTRUCT uncertainty-style artifact."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"{roi_name}_edited_rtstruct.dcm"
        export_rtstruct_uncertainty(
            out_path=out_path,
            binary_mask_3d=roi_mask.astype(bool),
            origin_xyz=study.geometry.origin,
            spacing_xyz=study.geometry.spacing,
            direction=study.geometry.direction.reshape(-1),
            frame_of_ref_uid=study.metadata.frame_of_reference_uid,
            study_instance_uid=study.metadata.study_instance_uid,
            patient_id=study.metadata.patient_id,
            patient_name=study.metadata.patient_name,
            roi_name=roi_name,
            ct_series_uid=study.metadata.ct_series_instance_uid,
            ct_sop_uids=study.metadata.ct_sop_instance_uids,
        )
        LOGGER.info("Exported edited ROI '%s' to %s", roi_name, out_path)
        return out_path
