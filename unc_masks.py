from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure


# ---------------------------------------------------------------------------
# Probability -> uncertainty heatmap
# ---------------------------------------------------------------------------

def uncertainty_from_p(
    p: np.ndarray,
    method: Literal["entropy", "variance", "passthrough"] = "entropy",
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert an uncertainty map u(x) ∈ [0,1] to an uncertainty heatmap ∈ [0,1].

    Use method="passthrough" when the input is already a meaningful uncertainty
    signal (e.g. a normalized surface-distance map) that should not be transformed.
    """
    p = np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0)

    if method == "passthrough":
        return p

    if method == "variance":
        return (4.0 * p * (1.0 - p)).astype(np.float32)

    if method == "entropy":
        p = np.clip(p, eps, 1.0 - eps)
        return (-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)).astype(np.float32)

    raise ValueError(f"Unknown method: {method!r}. Choose 'entropy', 'variance', or 'passthrough'.")


# ---------------------------------------------------------------------------
# Boundary / contour helpers
# ---------------------------------------------------------------------------

def boundary_voxels(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Return a bool mask of boundary voxels (mask minus its erosion)."""
    mask = mask.astype(bool)
    struct = generate_binary_structure(rank=3, connectivity=connectivity)
    return (mask & ~binary_erosion(mask, structure=struct, border_value=0))


def contour_heat(mask: np.ndarray, u: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Uncertainty values on boundary voxels only; zero elsewhere."""
    u_cont = np.zeros_like(u, dtype=np.float32)
    bnd = boundary_voxels(mask, connectivity)
    u_cont[bnd] = u[bnd]
    return u_cont


def contour_bin_threshold(
    mask: np.ndarray, u: np.ndarray, threshold: float = 0.5, connectivity: int = 1
) -> np.ndarray:
    """Binary uncertainty on contour voxels using a fixed threshold."""
    bnd = boundary_voxels(mask, connectivity)
    u_bin = np.zeros(u.shape, dtype=np.uint8)
    u_bin[bnd] = (u[bnd] >= threshold).astype(np.uint8)
    return u_bin


def contour_bin_top_fraction(
    mask: np.ndarray, u: np.ndarray, fraction: float = 0.25, connectivity: int = 1
) -> np.ndarray:
    """Binary uncertainty on contour voxels: top `fraction` most-uncertain boundary voxels."""
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in (0, 1)")

    bnd = boundary_voxels(mask, connectivity)
    vals = u[bnd]
    if vals.size == 0:
        raise RuntimeError("Boundary has zero voxels — mask may be empty.")

    threshold = float(np.quantile(vals, 1.0 - fraction))
    u_bin = np.zeros(u.shape, dtype=np.uint8)
    u_bin[bnd] = (u[bnd] >= threshold).astype(np.uint8)
    return u_bin


# ---------------------------------------------------------------------------
# Combined output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyOutputs:
    u_heat:       np.ndarray  # float32 (Z,Y,X) ∈ [0,1]  — full heatmap
    contour_heat: np.ndarray  # float32 (Z,Y,X)           — heatmap on boundary only
    contour_bin:  np.ndarray  # uint8   (Z,Y,X) 0/1       — binary on boundary only
    contour_mask: np.ndarray  # bool    (Z,Y,X)           — boundary voxels


def make_uncertainty_outputs(
    pmap: np.ndarray,
    mask_ai: np.ndarray,
    *,
    heat_method:          Literal["entropy", "variance", "passthrough"] = "entropy",
    contour_mode:         Literal["threshold", "top_fraction"] = "top_fraction",
    contour_threshold:    float = 0.5,
    contour_top_fraction: float = 0.25,
    contour_connectivity: int = 1,
) -> UncertaintyOutputs:
    """
    Produce contour uncertainty maps from a probability map.
    All outputs are aligned to (Z, Y, X).
    """
    mask_ai = mask_ai.astype(bool)
    u = uncertainty_from_p(pmap, method=heat_method)

    bnd = boundary_voxels(mask_ai, contour_connectivity)

    if contour_mode == "threshold":
        c_bin = contour_bin_threshold(mask_ai, u, contour_threshold, contour_connectivity)
    elif contour_mode == "top_fraction":
        c_bin = contour_bin_top_fraction(mask_ai, u, contour_top_fraction, contour_connectivity)
    else:
        raise ValueError(f"Unknown contour_mode: {contour_mode!r}")

    return UncertaintyOutputs(
        u_heat       = u,
        contour_heat = contour_heat(mask_ai, u, contour_connectivity),
        contour_bin  = c_bin,
        contour_mask = bnd,
    )