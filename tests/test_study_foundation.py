from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from export_service import ExportService
from models import DicomStudyMetadata, LoadedStudy
from roi_extraction import VolumeGeometry, _load_ct_volume, _load_geometry
from study_loader import load_study


def _write_ct_slice(path: Path, z_mm: float, value: int, *, sop_uid: str | None = None) -> str:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "CT"
    ds.PatientID = "P001"
    ds.PatientName = "Test^Patient"
    ds.StudyInstanceUID = "1.2.3"
    ds.SeriesInstanceUID = "1.2.3.4"
    ds.FrameOfReferenceUID = "9.9.9"
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(z_mm)]
    ds.PixelSpacing = [1.0, 1.0]
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = (np.ones((4, 4), dtype=np.int16) * value).tobytes()
    pydicom.dcmwrite(str(path), ds, write_like_original=False)
    return str(ds.SOPInstanceUID)


def test_slice_sorting_consistency(tmp_path: Path) -> None:
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    _write_ct_slice(ct_dir / "slice2.dcm", z_mm=2.0, value=20)
    _write_ct_slice(ct_dir / "slice0.dcm", z_mm=0.0, value=0)
    _write_ct_slice(ct_dir / "slice1.dcm", z_mm=1.0, value=10)

    ct_volume = _load_ct_volume(ct_dir)
    geometry = _load_geometry(ct_dir)

    assert ct_volume.shape == (3, 4, 4)
    assert [int(ct_volume[z, 0, 0]) for z in range(3)] == [0, 10, 20]
    assert np.isclose(geometry.spacing[2], 1.0)


def test_geometry_conversion_sanity() -> None:
    geom = VolumeGeometry(
        origin=np.array([10.0, 20.0, 30.0]),
        spacing=np.array([2.0, 3.0, 4.0]),
        direction=np.eye(3),
    )
    ijk = np.array([1.5, 2.0, 0.5])
    xyz = geom.voxel_to_mm(ijk)
    assert np.allclose(geom.mm_to_voxel(xyz), ijk)


def test_load_study_metadata_and_shape(tmp_path: Path, monkeypatch) -> None:
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    _write_ct_slice(ct_dir / "a.dcm", z_mm=0.0, value=1)
    _write_ct_slice(ct_dir / "b.dcm", z_mm=1.0, value=2)

    rtstruct_path = tmp_path / "rtstruct.dcm"
    ds = Dataset()
    ds.SOPInstanceUID = "7.7.7"
    pydicom.dcmwrite(str(rtstruct_path), ds)

    class _DummyRt:
        def get_roi_names(self):
            return ["GTV"]

        def get_roi_mask_by_name(self, name: str):
            assert name == "GTV"
            return np.transpose(np.ones((2, 4, 4), dtype=bool), (1, 2, 0))

    monkeypatch.setattr("study_loader.RTStructBuilder.create_from", lambda **_: _DummyRt())

    study = load_study(ct_dir, rtstruct_path)
    assert study.ct_volume.shape == (2, 4, 4)
    assert study.roi_masks["GTV"].shape == study.ct_volume.shape
    assert study.metadata.rtstruct_sop_instance_uid == "7.7.7"


def test_export_service_minimal(tmp_path: Path) -> None:
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), direction=np.eye(3))
    study = LoadedStudy(
        ct_volume=np.zeros((2, 4, 4), dtype=np.int16),
        geometry=geom,
        roi_masks={"GTV": np.zeros((2, 4, 4), dtype=bool)},
        roi_names=["GTV"],
        ct_folder=tmp_path,
        rtstruct_path=tmp_path / "in_rtstruct.dcm",
        metadata=DicomStudyMetadata(
            patient_id="P001",
            patient_name="Test^Patient",
            study_instance_uid="1.2.3",
            ct_series_instance_uid="1.2.3.4",
            rtstruct_sop_instance_uid="7.7.7",
            frame_of_reference_uid="9.9.9",
            ct_sop_instance_uids=["1.2.3.4.5"],
        ),
    )
    out = ExportService(tmp_path / "out").export_edited_roi(study, "GTV", np.zeros((2, 4, 4), dtype=bool))
    assert out.exists()
    exported = pydicom.dcmread(str(out), stop_before_pixels=True)
    assert exported.Modality == "RTSTRUCT"
