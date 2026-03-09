from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roi_extraction import VolumeGeometry


@dataclass(frozen=True)
class DicomStudyMetadata:
    """Patient/study/series identifiers needed by the desktop application."""

    patient_id: str
    patient_name: str
    study_instance_uid: str
    ct_series_instance_uid: str
    rtstruct_sop_instance_uid: str
    frame_of_reference_uid: str
    ct_sop_instance_uids: list[str] = field(default_factory=list)


@dataclass
class RoiLayer:
    """A named ROI mask layer in ``(Z, Y, X)`` layout."""

    name: str
    mask: np.ndarray


@dataclass
class LoadedStudy:
    """Unified in-memory study representation for GUI and editing workflows."""

    ct_volume: np.ndarray
    geometry: VolumeGeometry
    roi_masks: dict[str, np.ndarray]
    roi_names: list[str]
    ct_folder: Path
    rtstruct_path: Path
    metadata: DicomStudyMetadata
