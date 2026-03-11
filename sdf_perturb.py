from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    binary_fill_holes,
    label,
)


# ----------------------------
# Core: SDF + perturbation
# ----------------------------

def signed_distance_mm(mask_zyx: np.ndarray, spacing_zyx_mm: Tuple[float, float, float]) -> np.ndarray:
    """
    Compute signed distance field in mm.
    Convention:
        sdf < 0 inside mask, sdf > 0 outside, sdf == 0 near boundary.

    Args:
        mask_zyx: bool array (Z,Y,X)
        spacing_zyx_mm: (dz, dy, dx) in mm

    Returns:
        sdf_mm: float32 array (Z,Y,X)
    """
    mask = mask_zyx.astype(bool)
    # distance to nearest background for inside pixels
    dist_in = distance_transform_edt(mask, sampling=spacing_zyx_mm)
    # distance to nearest foreground for outside pixels
    dist_out = distance_transform_edt(~mask, sampling=spacing_zyx_mm)
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest connected component (6-connectivity in 3D)."""
    lab, n = label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background
    largest = counts.argmax()
    return (lab == largest)


@dataclass(frozen=True)
class PerturbParams:
    amp_mm: float = 5.0              # maximum-ish deformation scale in mm
    smooth_mm: float = 4.0           # smoothness of noise field (Gaussian sigma, mm)
    band_mm: float = 6.0             # how far from boundary perturbation acts (mm)
    bias_mm: float = 0.0             # systematic bias (+ expands, - shrinks), mm


def perturb_mask_via_sdf(
    mask_gt: np.ndarray,
    spacing_zyx_mm: Tuple[float, float, float],
    params: PerturbParams,
    seed: int = 0,
    *,
    postprocess: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a perturbed mask by adding a smooth boundary-local field to the SDF.

    Returns:
        mask_ai: bool (Z,Y,X)
        sdf_mm: float32 (Z,Y,X) original sdf
        delta_mm: float32 (Z,Y,X) added perturbation in mm
    """
    # Work on a copy to avoid mutating caller-provided arrays
    mask_gt = mask_gt.copy()
    rng = np.random.default_rng(seed)

    # Compute signed distance field from a copy
    sdf_mm = signed_distance_mm(mask_gt, spacing_zyx_mm)

    # Smooth random field in voxel units (convert smooth_mm -> sigma_vox)
    dz, dy, dx = spacing_zyx_mm
    sigma_vox = (params.smooth_mm / dz, params.smooth_mm / dy, params.smooth_mm / dx)

    noise = rng.standard_normal(size=mask_gt.shape).astype(np.float32)
    noise_smooth = gaussian_filter(noise, sigma=sigma_vox, mode="reflect")

    # Normalize noise to roughly [-1, 1] robustly (avoid outliers)
    p95 = np.percentile(np.abs(noise_smooth), 95)
    if p95 > 0:
        noise_smooth = noise_smooth / p95
    noise_smooth = np.clip(noise_smooth, -1.0, 1.0)

    # Gate perturbation to a band around the boundary
    # gate ~ 1 near sdf=0, decays with distance from boundary
    gate = np.exp(-(sdf_mm ** 2) / (2.0 * (params.band_mm ** 2))).astype(np.float32)

    delta_mm = (params.bias_mm + params.amp_mm * noise_smooth) * gate

    sdf_pert = sdf_mm + delta_mm
    mask_ai = sdf_pert <= 0.0

    if postprocess:
        # postprocess returns new arrays; ensure originals are not modified
        mask_ai = binary_fill_holes(mask_ai)
        mask_ai = _keep_largest_component(mask_ai)

    return mask_ai.astype(bool), sdf_mm, delta_mm.astype(np.float32)


# ----------------------------
# Optional: pseudo-ensemble -> p-map
# ----------------------------

def pseudo_ensemble_probability(
    mask_gt: np.ndarray,
    spacing_zyx_mm: Tuple[float, float, float],
    base_params: PerturbParams,
    n: int = 20,
    seed: int = 0,
    amp_jitter: float = 0.35,
    bias_jitter_mm: float = 0.5,
) -> np.ndarray:
    """
    Generate N perturbed masks around GT and return p(x)=mean(mask_n).
    Useful later for heatmap uncertainty.

    amp_jitter: relative std-dev for amplitude (e.g. 0.35 means ~35% jitter)
    bias_jitter_mm: add small random global bias per sample to simulate systematic shifts
    """
    # Work on a copy to avoid mutating the caller's mask
    mask_gt = mask_gt.copy()
    rng = np.random.default_rng(seed)
    acc = np.zeros(mask_gt.shape, dtype=np.float32)

    for i in range(n):
        amp = max(0.1, float(base_params.amp_mm * (1.0 + amp_jitter * rng.standard_normal())))
        bias = float(base_params.bias_mm + bias_jitter_mm * rng.standard_normal())
        params_i = PerturbParams(
            amp_mm=amp,
            smooth_mm=base_params.smooth_mm,
            band_mm=base_params.band_mm,
            bias_mm=bias,
        )
        mask_i, _, _ = perturb_mask_via_sdf(
            mask_gt, spacing_zyx_mm, params_i, seed=int(seed + 1000 + i), postprocess=True
        )
        acc += mask_i.astype(np.float32)

    return (acc / float(n)).astype(np.float32)


# ----------------------------
# CLI
# ----------------------------

# def main() -> int:
#     parser = argparse.ArgumentParser(description="SDF-based perturbation of an ROI mask (mm-aware).")
#     parser.add_argument("--in-npz", required=True, help="Input .npz from ROI extraction (ct_volume, roi_mask, spacing, ...)")
#     parser.add_argument("--out-npz", required=True, help="Output .npz with mask_ai (+ sdf/delta optional)")
#     parser.add_argument("--seed", type=int, default=0)

#     parser.add_argument("--amp-mm", type=float, default=4.0)
#     parser.add_argument("--smooth-mm", type=float, default=3.0)
#     parser.add_argument("--band-mm", type=float, default=5.0)
#     parser.add_argument("--bias-mm", type=float, default=0.0)

#     parser.add_argument("--make-pmap", action="store_true", help="Also compute pseudo-ensemble probability map p(x)")
#     parser.add_argument("--n-ens", type=int, default=20)

#     args = parser.parse_args()

#     data = np.load(args.in_npz, allow_pickle=True)
#     mask_gt = data["roi_mask"].astype(bool)

#     # IMPORTANT: Your saved 'spacing' must match the order expected here.
#     # This script expects spacing_zyx_mm = (dz, dy, dx).
#     # If your extractor stores spacing as (dx, dy, dz), reorder it here.
#     spacing = data["spacing"].astype(np.float32)

#     if spacing.shape != (3,):
#         raise RuntimeError(f"Expected spacing shape (3,), got {spacing.shape}")

#     # Heuristic: if you stored (dx,dy,dz) (common), convert to (dz,dy,dx)
#     # For CT, dz is often >= dx,dy. If spacing[2] looks like dz (e.g. 1.25) and spacing[0] looks like dx (~0.7),
#     # then spacing is likely (dx,dy,dz) and we should reorder.
#     dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
#     if dz >= dx and dz >= dy:
#         spacing_zyx = (dz, dy, dx)  # (dz,dy,dx)
#     else:
#         # already maybe (dz,dy,dx)
#         spacing_zyx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

#     params = PerturbParams(
#         amp_mm=float(args.amp_mm),
#         smooth_mm=float(args.smooth_mm),
#         band_mm=float(args.band_mm),
#         bias_mm=float(args.bias_mm),
#     )

#     mask_ai, sdf_mm, delta_mm = perturb_mask_via_sdf(
#         mask_gt, spacing_zyx, params, seed=args.seed, postprocess=True
#     )

#     out = {
#         "mask_gt": mask_gt.astype(np.uint8),
#         "mask_ai": mask_ai.astype(np.uint8),
#         "spacing_zyx_mm": np.array(spacing_zyx, dtype=np.float32),
#         "params": np.array([args.amp_mm, args.smooth_mm, args.band_mm, args.bias_mm], dtype=np.float32),
#         "seed": np.int32(args.seed),
#         "sdf_mm": sdf_mm,
#         "delta_mm": delta_mm,
#     }

#     if args.make_pmap:
#         pmap = pseudo_ensemble_probability(
#             mask_gt, spacing_zyx, params, n=int(args.n_ens), seed=args.seed
#         )
#         out["pmap"] = pmap

#     Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
#     np.savez_compressed(args.out_npz, **out)

#     print("Saved:", args.out_npz)
#     print("mask_gt voxels:", int(mask_gt.sum()), "mask_ai voxels:", int(mask_ai.sum()))
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())