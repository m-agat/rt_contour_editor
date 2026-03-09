from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydicom
from rt_utils import RTStructBuilder

@dataclass
class CtRtstructMatch:
    ct_folder: Path
    ct_series_uid: str
    rtstruct_path: Path
    roi_names: list[str]


def _read_dicom(path: Path) -> Optional[pydicom.Dataset]:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _index_ct_series(visit_dir: Path) -> dict[str, Path]:
    """Return {SeriesInstanceUID: folder} for CT series found in immediate subfolders."""
    ct_series: dict[str, Path] = {}

    for folder in [p for p in visit_dir.iterdir() if p.is_dir()]:
        uid_counts: dict[str, int] = {}

        for f in list(folder.rglob("*"))[:2000]:
            ds = _read_dicom(f)
            if ds is None or getattr(ds, "Modality", "").upper() != "CT":
                continue
            uid = str(ds.SeriesInstanceUID) if hasattr(ds, "SeriesInstanceUID") else None
            if uid:
                uid_counts[uid] = uid_counts.get(uid, 0) + 1

        if uid_counts:
            best_uid = max(uid_counts, key=uid_counts.__getitem__)
            ct_series[best_uid] = folder

    return ct_series


def _get_referenced_series_uid(rtstruct_path: Path) -> Optional[str]:
    ds = _read_dicom(rtstruct_path)
    if ds is None:
        return None
    try:
        return str(
            ds.ReferencedFrameOfReferenceSequence[0]
            .RTReferencedStudySequence[0]
            .RTReferencedSeriesSequence[0]
            .SeriesInstanceUID
        )
    except Exception:
        return None


def _find_rtstruct_files(visit_dir: Path) -> list[Path]:
    return [
        p for p in visit_dir.rglob("*")
        if p.is_file()
        and (ds := _read_dicom(p)) is not None
        and getattr(ds, "Modality", "").upper() == "RTSTRUCT"
    ]


def _score_candidate(roi_names: list[str], preferred: list[str]) -> tuple[int, int]:
    """Score by (# preferred ROI substrings matched, total ROI count)."""
    if not preferred:
        return (0, len(roi_names))
    roi_lower = [r.lower() for r in roi_names]
    hits = sum(1 for p in preferred if any(p in r for r in roi_lower))
    return (hits, len(roi_names))


def find_ct_and_rtstruct(
    visit_dir: Path,
    prefer_roi_substrings: Optional[list[str]] = None,
) -> CtRtstructMatch:
    """
    Find the CT series and best-matching RTSTRUCT in a visit folder.

    Args:
        visit_dir: path to a single radiotherapy visit folder.
        prefer_roi_substrings: ROI name substrings to prefer (e.g. ["GTV", "PTV"]).

    Returns:
        CtRtstructMatch with CT folder, series UID, RTSTRUCT path, and ROI names.
    """
    visit_dir = Path(visit_dir)
    ct_series = _index_ct_series(visit_dir)
    if not ct_series:
        raise RuntimeError(f"No CT series found under: {visit_dir}")

    rtstruct_files = _find_rtstruct_files(visit_dir)
    if not rtstruct_files:
        raise RuntimeError(f"No RTSTRUCT files found under: {visit_dir}")

    preferred = [s.lower() for s in (prefer_roi_substrings or []) if s.strip()]

    # Build (rtstruct_path, ct_folder, ct_uid, roi_names) for each RT that references a CT
    candidates = []
    for rt_path in rtstruct_files:
        ref_uid = _get_referenced_series_uid(rt_path)
        ct_folder = ct_series.get(ref_uid) if ref_uid else None
        if ct_folder is None:
            continue

        try:
            roi_names = RTStructBuilder.create_from(
                dicom_series_path=str(ct_folder),
                rt_struct_path=str(rt_path),
            ).get_roi_names()
        except Exception:
            roi_names = []

        candidates.append((rt_path, ct_folder, ref_uid, roi_names))

    if not candidates:
        raise RuntimeError(f"No RTSTRUCT matched to a CT series in: {visit_dir}")

    best = max(candidates, key=lambda c: _score_candidate(c[3], preferred))
    rt_path, ct_folder, ct_uid, roi_names = best

    return CtRtstructMatch(
        ct_folder=ct_folder,
        ct_series_uid=ct_uid,
        rtstruct_path=rt_path,
        roi_names=roi_names,
    )


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--visit", required=True)
#     parser.add_argument("--prefer", action="append", default=[])
#     args = parser.parse_args()

#     match = find_ct_and_rtstruct(Path(args.visit), prefer_roi_substrings=args.prefer)
#     print("CT folder     :", match.ct_folder)
#     print("CT series UID :", match.ct_series_uid)
#     print("RTSTRUCT      :", match.rtstruct_path)
#     print("ROIs          :", match.roi_names[:30])