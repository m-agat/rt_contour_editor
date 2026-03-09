from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom                          
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, distance_transform_edt

from ct_rtstruct_matching import find_ct_and_rtstruct, CtRtstructMatch
from roi_extraction import extract_roi, RoiExtractionResult
from sdf_perturb import PerturbParams, perturb_mask_via_sdf
from unc_masks import make_uncertainty_outputs, UncertaintyOutputs
from dicom_export import export_uncertainty_to_dicom  


# Root folder containing all visit subfolders 
RT_DATA_DIR = Path(__file__).resolve().parent / "rt_data"

PREFER = ["GTV", "PTV"]

# -----------------------------
# Visualization constants
# -----------------------------
UNC_CMAP = "magma"
HEAT_ALPHA = 0.90

GT_COLOR = (0.4, 1.0, 0.2)      # green
AI_COLOR = (0.97, 0.97, 0.97)   # near-white
BIN_COLOR = (1.0, 0.75, 0.0)    # orange
BIN_ALPHA = 0.95

GT_LW = 1.3
AI_LW = 2.0

# Visibility
CROP_MARGIN_PX = 50
AI_NEIGHBOR_DILATE = 4
BIN_CONTOUR_DISPLAY_THICKEN = 2
DIFF_ALPHA = 0.35

# --- Contour uncertainty ---
CONTOUR_TOP_FRACTION = 0.25    # top 25% uncertainty on contour pixels only

# If you want clinician stimuli (no GT), set to False:
SHOW_GT_PANELS = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_roi(roi_names: list[str], prefer: list[str]) -> str:
    for p in prefer:
        for name in roi_names:
            if p.lower() in name.lower():
                return name
    if not roi_names:
        raise RuntimeError("RTSTRUCT contains no ROIs.")
    return roi_names[0]


def spacing_zyx(result: RoiExtractionResult) -> tuple[float, float, float]:
    dx, dy, dz = result.geometry.spacing
    return (float(dz), float(dy), float(dx))


def spacing_yx_mm(result: RoiExtractionResult) -> tuple[float, float]:
    """(dy, dx) in mm for a Z-slice."""
    dx, dy, _dz = result.geometry.spacing
    return float(dy), float(dx)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print("Saved:", path)


def common_arrays(match: CtRtstructMatch, result: RoiExtractionResult) -> dict:
    return dict(
        ct_volume=result.ct_volume,
        roi_mask=result.roi_mask,
        origin=result.geometry.origin,
        spacing=result.geometry.spacing,
        direction=result.geometry.direction,
        roi_name=result.roi_name,
        ct_folder=str(match.ct_folder),
        rtstruct_path=str(match.rtstruct_path),
        ct_series_uid=match.ct_series_uid,
    )


def _bbox_from_mask(mask2d: np.ndarray, margin: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask2d)
    if len(xs) == 0 or len(ys) == 0:
        return slice(0, mask2d.shape[0]), slice(0, mask2d.shape[1])

    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask2d.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask2d.shape[1])
    return slice(y0, y1), slice(x0, x1)


def _crop2d(arr2d: np.ndarray, ys: slice, xs: slice) -> np.ndarray:
    return arr2d[ys, xs]


def _show_ct(ax, ct2d: np.ndarray, vmin: float, vmax: float) -> None:
    ax.imshow(ct2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")


def _colorbar(fig, mappable, ax, label: str) -> None:
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04).set_label(label)


def _solid_overlay(ax, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = (*color, alpha)
    ax.imshow(overlay)


def _boundary_pixels(mask: np.ndarray, struct2d) -> np.ndarray:
    """1px boundary pixels."""
    return mask ^ binary_erosion(mask, structure=struct2d, iterations=1)


def uncertainty_from_dist(dist2d: np.ndarray) -> np.ndarray:
    """Pass through the normalized distance map as the uncertainty signal.
    Values are already in [0,1] where 1 = furthest from GT boundary."""
    return np.clip(dist2d, 0.0, 1.0).astype(np.float32)


def make_binary_contour_from_pmap(
    pmap2d: np.ndarray,
    ai_mask2d: np.ndarray,
    struct2d,
    top_fraction: float = CONTOUR_TOP_FRACTION,
) -> np.ndarray:
    """
    Contour (segments): top_fraction of most-distant (from GT) boundary pixels.
    """
    u = uncertainty_from_dist(pmap2d)
    bnd = _boundary_pixels(ai_mask2d, struct2d)
    vals = u[bnd]
    if vals.size == 0:
        return np.zeros_like(ai_mask2d, dtype=bool)
    q = float(np.quantile(vals, 1.0 - float(top_fraction)))
    return bnd & (u >= q)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_match(visit_dir: Path) -> CtRtstructMatch:
    match = find_ct_and_rtstruct(visit_dir, prefer_roi_substrings=PREFER)
    print("CT folder     :", match.ct_folder)
    print("RTSTRUCT      :", match.rtstruct_path)
    print("ROIs          :", match.roi_names[:10])
    return match


def run_extraction(match: CtRtstructMatch, roi_name: str) -> RoiExtractionResult:
    result = extract_roi(match.ct_folder, match.rtstruct_path, roi_name)
    vol_cm3 = result.roi_mask.sum() * np.prod(result.geometry.spacing) / 1000.0
    print(f"ROI           : {result.roi_name}")
    print(f"CT shape      : {result.ct_volume.shape}  (Z, Y, X)")
    print(f"Volume        : {vol_cm3:.2f} cm3")
    print(f"Spacing (mm)  : {result.geometry.spacing}")
    return result


def run_perturbation(
    result: RoiExtractionResult,
    params: PerturbParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single SDF perturbation of GT -> mask_ai (simulated AI contour)."""
    mask_ai, sdf_mm, delta_mm = perturb_mask_via_sdf(
        result.roi_mask, spacing_zyx(result), params, seed=0, postprocess=True
    )
    print(f"mask_ai voxels : {mask_ai.sum()}  (GT: {result.roi_mask.sum()})")
    return mask_ai, sdf_mm, delta_mm


def compute_surface_distance_pmap(
    mask_ai: np.ndarray,
    mask_gt: np.ndarray,
    sp_zyx: tuple[float, float, float],
) -> np.ndarray:
    """
    For every AI boundary voxel, compute its Euclidean distance (mm) to the
    nearest GT boundary voxel. Non-boundary voxels get distance 0.

    Returns a float32 array normalized to [0, 1] by the 99th percentile of
    non-zero AI-boundary distances (robust to outliers).
    """
    struct3d = generate_binary_structure(rank=3, connectivity=1)

    gt_bnd = mask_gt.astype(bool) & ~binary_erosion(
        mask_gt.astype(bool), structure=struct3d, border_value=0
    )
    # EDT on ~gt_bnd: distance from every voxel to the nearest GT boundary voxel
    dist_to_gt_surface = distance_transform_edt(~gt_bnd, sampling=sp_zyx).astype(np.float32)

    ai_bnd = mask_ai.astype(bool) & ~binary_erosion(
        mask_ai.astype(bool), structure=struct3d, border_value=0
    )

    # Uncertainty map: distance-to-GT-surface, but only on AI boundary voxels
    dist_map = np.zeros_like(dist_to_gt_surface)
    dist_map[ai_bnd] = dist_to_gt_surface[ai_bnd]

    # Normalize to [0, 1]
    vals = dist_map[ai_bnd]
    scale = float(max(np.percentile(vals, 99), 1e-3)) if vals.size > 0 else 1.0
    pmap = np.clip(dist_map / scale, 0.0, 1.0).astype(np.float32)

    print(f"AI boundary voxels : {int(ai_bnd.sum())}  |  max dist : {vals.max():.2f} mm  |  scale : {scale:.2f} mm")
    return pmap


def compute_pmap(
    mask_ai: np.ndarray,
    result: RoiExtractionResult,
    params: PerturbParams,
    n: int = 10,
) -> tuple[np.ndarray, np.ndarray, UncertaintyOutputs]:
    """
    Uncertainty = distance from each AI boundary voxel to the nearest GT boundary
    voxel (mm), normalized to [0,1]. Top 25% most-distant AI boundary voxels are
    flagged as uncertain in the binary contour map.
    """
    sp_zyx = spacing_zyx(result)
    pmap = compute_surface_distance_pmap(mask_ai, result.roi_mask, sp_zyx)

    mask_mean = mask_ai.astype(bool)

    unc = make_uncertainty_outputs(
        pmap, mask_ai,
        heat_method="passthrough",
        contour_mode="top_fraction",
        contour_top_fraction=0.25,
    )
    print(f"mask_mean voxels : {mask_mean.sum()}  (mask_ai: {mask_ai.sum()})")
    return pmap, mask_mean, unc


# ---------------------------------------------------------------------------
# DICOM UID extraction from CT folder
# ---------------------------------------------------------------------------

def _read_ct_dicom_metadata(ct_folder: Path) -> dict:
    """
    Read the first CT DICOM slice in ct_folder and extract the UIDs needed
    to anchor the exported RT Dose / RTSTRUCT to the correct study.

    Returns a dict with keys:
        frame_of_ref_uid, study_instance_uid, series_instance_uid,
        patient_id, patient_name, sop_uids (list[str])
    """
    dcm_files = sorted(ct_folder.glob("*.dcm"))
    if not dcm_files:
        # Some datasets use no extension
        dcm_files = sorted(
            f for f in ct_folder.iterdir()
            if f.is_file() and not f.suffix.lower() in {".txt", ".xml", ".json"}
        )

    meta: dict = {
        "frame_of_ref_uid": "1.2.3.UNKNOWN",
        "study_instance_uid": "1.2.3.UNKNOWN",
        "series_instance_uid": "1.2.3.UNKNOWN",
        "patient_id": "UNKNOWN",
        "patient_name": "UNKNOWN",
        "sop_uids": [],
    }

    sop_uids: list[str] = []
    first = True
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue

        sop = getattr(ds, "SOPInstanceUID", None)
        if sop:
            sop_uids.append(str(sop))

        if first:
            meta["frame_of_ref_uid"] = str(
                getattr(ds, "FrameOfReferenceUID", meta["frame_of_ref_uid"])
            )
            meta["study_instance_uid"] = str(
                getattr(ds, "StudyInstanceUID", meta["study_instance_uid"])
            )
            meta["series_instance_uid"] = str(
                getattr(ds, "SeriesInstanceUID", meta["series_instance_uid"])
            )
            meta["patient_id"] = str(
                getattr(ds, "PatientID", "UNKNOWN")
            )
            meta["patient_name"] = str(
                getattr(ds, "PatientName", "UNKNOWN")
            )
            first = False

    meta["sop_uids"] = sop_uids
    return meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_comparison_plot(
    out_path: Path,
    result: RoiExtractionResult,
    mask_ai: np.ndarray,
    pmap: np.ndarray,
    unc: UncertaintyOutputs
) -> None:
    """
    1x4 grid (zoomed/cropped for visibility):
        [GT contour] [GT vs AI] [Uncertainty on contour (continuous)] [Uncertain contour segments (binary)]
    """
    counts = unc.contour_bin.sum(axis=(1, 2))
    if counts.max() > 0:
        z = int(np.argmax(counts))
    else:
        z = int(np.argmax(result.roi_mask.sum(axis=(1, 2))))

    ct = result.ct_volume[z]
    vmin = float(np.percentile(ct, 1))
    vmax = float(np.percentile(ct, 99))

    gt_s = result.roi_mask[z].astype(bool)
    ai_s = mask_ai[z].astype(bool)
    p_s = pmap[z].astype(np.float32)

    struct2d = generate_binary_structure(2, 1)

    ref_mask = ai_s if ai_s.any() else gt_s
    ys, xs = _bbox_from_mask(ref_mask, margin=CROP_MARGIN_PX)

    ct_c = _crop2d(ct, ys, xs)
    gt_c = _crop2d(gt_s, ys, xs)
    ai_c = _crop2d(ai_s, ys, xs)
    p_c = _crop2d(p_s, ys, xs)

    u_c = uncertainty_from_dist(p_c)

    cb_c = make_binary_contour_from_pmap(
        pmap2d=p_c, ai_mask2d=ai_c, struct2d=struct2d,
        top_fraction=CONTOUR_TOP_FRACTION,
    )
    cb_show = binary_dilation(cb_c, structure=struct2d, iterations=BIN_CONTOUR_DISPLAY_THICKEN)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{result.roi_name} — slice {z}", fontsize=13)

    # 1) GT contour
    ax = axes[0]
    _show_ct(ax, ct_c, vmin, vmax)
    if SHOW_GT_PANELS:
        ax.contour(gt_c, levels=[0.5], colors=[GT_COLOR], linewidths=2.0)
        ax.set_title("GT contour")
    else:
        ax.contour(ai_c, levels=[0.5], colors=[AI_COLOR], linewidths=AI_LW)
        ax.set_title("AI contour")

    # 2) GT vs AI
    ax = axes[1]
    _show_ct(ax, ct_c, vmin, vmax)
    if SHOW_GT_PANELS:
        ax.contour(gt_c, levels=[0.5], colors=[GT_COLOR], linewidths=GT_LW)
    ax.contour(ai_c, levels=[0.5], colors=[AI_COLOR], linewidths=AI_LW)
    ax.set_title("GT (green) vs AI (white)")

    # 3) Uncertainty on contour (continuous)
    ax = axes[2]
    _show_ct(ax, ct_c, vmin, vmax)
    bnd = _boundary_pixels(ai_c, struct2d)
    contour_u = np.ma.masked_where(~bnd, u_c)
    im = ax.imshow(contour_u, cmap=UNC_CMAP, vmin=0, vmax=1, alpha=1.0)
    _colorbar(fig, im, ax, "uncertainty")
    ax.set_title("Uncertainty on contour (continuous)")

    # 4) Uncertainty on contour (binary)
    ax = axes[3]
    _show_ct(ax, ct_c, vmin, vmax)
    rgba = np.zeros((*cb_show.shape, 4), dtype=np.float32)
    rgba[cb_show] = (*BIN_COLOR, BIN_ALPHA)
    ax.imshow(rgba, interpolation="none")
    ax.contour(ai_c, levels=[0.5], colors=[AI_COLOR], linewidths=AI_LW)
    ax.legend(
        handles=[Patch(
            facecolor=BIN_COLOR,
            edgecolor="none",
            label=f"top {int(CONTOUR_TOP_FRACTION*100)}% boundary U",
        )],
        loc="lower left",
    )
    ax.set_title("Uncertain contour segments (binary)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

    bnd = _boundary_pixels(ai_c, struct2d)
    contour_heat_c = (u_c * bnd).astype(np.float32)

    save_npz(
        out_path.with_suffix(".derived_unc.npz"),
        z=np.array([z], dtype=np.int32),
        crop_y0=np.array([ys.start], dtype=np.int32),
        crop_y1=np.array([ys.stop], dtype=np.int32),
        crop_x0=np.array([xs.start], dtype=np.int32),
        crop_x1=np.array([xs.stop], dtype=np.int32),
        contour_heat=contour_heat_c,
        contour_bin=cb_c.astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not RT_DATA_DIR.exists():
        print("rt_data folder not found:", RT_DATA_DIR)
        return 1

    params = PerturbParams()

    visits = sorted([p for p in RT_DATA_DIR.iterdir() if p.is_dir()])
    if not visits:
        print("No visit subfolders found in:", RT_DATA_DIR)
        return 1

    for visit_dir in visits:
        print(f"\n--- Processing visit: {visit_dir.name} ---")
        out_dir = Path(__file__).resolve().parent / "out" / visit_dir.name
        try:
            match = run_match(visit_dir)
            roi_name = select_roi(match.roi_names, PREFER)
            print(f"Selected ROI  : {roi_name}")

            result = run_extraction(match, roi_name)
            save_npz(out_dir / f"{roi_name}.npz", **common_arrays(match, result))

            mask_ai, sdf_mm, delta_mm = run_perturbation(result, params)
            save_npz(
                out_dir / f"{roi_name}_perturbed.npz",
                **common_arrays(match, result),
                mask_ai=mask_ai.astype(np.uint8),
                sdf_mm=sdf_mm,
                delta_mm=delta_mm,
            )

            pmap, mask_mean, unc = compute_pmap(mask_ai, result, params, n=10)
            save_npz(
                out_dir / f"{roi_name}_.npz",
                **common_arrays(match, result),
                mask_ai=mask_ai.astype(np.uint8),
                pmap=pmap,
                mask_mean=mask_mean.astype(np.uint8),
                contour_heat=unc.contour_heat,
                contour_bin=unc.contour_bin,
                contour_mask=unc.contour_mask.astype(np.uint8),
                sdf_mm=sdf_mm,
                delta_mm=delta_mm,
            )

            save_comparison_plot(
                out_path=out_dir / f"{roi_name}_comparison.png",
                result=result,
                mask_ai=mask_ai,
                pmap=pmap,
                unc=unc,
            )

            # ----------------------------------------------------------------
            # NEW: Export uncertainty maps to DICOM
            # ----------------------------------------------------------------
            print(f"\nExporting DICOM uncertainty maps for {roi_name} ...")
            ct_meta = _read_ct_dicom_metadata(match.ct_folder)

            export_uncertainty_to_dicom(
                out_dir=out_dir / "dicom",
                roi_name=roi_name,
                pmap=pmap,
                contour_bin=unc.contour_bin,
                origin=result.geometry.origin,       # (x, y, z) mm
                spacing=result.geometry.spacing,     # (dx, dy, dz) mm
                direction=result.geometry.direction,
                frame_of_ref_uid=ct_meta["frame_of_ref_uid"],
                study_instance_uid=ct_meta["study_instance_uid"],
                patient_id=ct_meta["patient_id"],
                patient_name=ct_meta["patient_name"],
                ct_series_uid=ct_meta["series_instance_uid"],
                ct_sop_uids=ct_meta["sop_uids"],
            )
            # ----------------------------------------------------------------

        except Exception as exc:
            print(f"Error processing {visit_dir.name}: {exc}")
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())