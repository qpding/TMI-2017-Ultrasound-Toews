"""Rigid registration utility inspired by autocalibration.cpp.

The original implementation optimized a rigid ultrasound-to-tracker
transform. This Python version loads corresponding point sets from
multiple ``*.non-linear.matches.X`` / ``*.non-linear.matches.Y`` files
and estimates a best-fit rigid (or similarity) transform with SVD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def load_point_pairs(prefixes: Iterable[Path], cols: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for prefix in prefixes:
        x_path = prefix.with_suffix(".non-linear.matches.X")
        y_path = prefix.with_suffix(".non-linear.matches.Y")
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing matching pair for prefix '{prefix}'.")

        x = np.loadtxt(x_path)
        y = np.loadtxt(y_path)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Row mismatch between {x_path} and {y_path}.")
        xs.append(x[:, cols])
        ys.append(y[:, cols])
        print(f"Loaded {x.shape[0]} correspondences from {x_path} / {y_path}")

    return np.vstack(xs), np.vstack(ys)


def umeyama_alignment(source: np.ndarray, target: np.ndarray, allow_scaling: bool) -> Tuple[np.ndarray, float]:
    """Compute the transformation that aligns source->target using Umeyama's method."""

    if source.shape != target.shape:
        raise ValueError("Source and target must share the same shape")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Point arrays must be shaped (N, 3)")

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_centered = source - mu_src
    tgt_centered = target - mu_tgt

    covariance = tgt_centered.T @ src_centered / source.shape[0]
    U, S, Vt = np.linalg.svd(covariance)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    if allow_scaling:
        var_src = np.sum(src_centered ** 2) / source.shape[0]
        scale = np.sum(S) / var_src
    else:
        scale = 1.0

    t = mu_tgt - scale * R @ mu_src
    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t

    residuals = target - (scale * (R @ source.T).T + t)
    rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    return transform, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a rigid transform from point correspondences.")
    parser.add_argument("prefixes", nargs="+", type=Path, help="Basenames used to find *.non-linear.matches.X/Y files.")
    parser.add_argument("--cols", type=str, default="0,1,2", help="Comma-separated column indices to read as XYZ (0-based).")
    parser.add_argument("--allow-scale", action="store_true", help="Estimate an isotropic scale in addition to rotation/translation.")
    parser.add_argument("--output", type=Path, default=Path("autocalibration_result.json"), help="Where to write the resulting transform.")
    args = parser.parse_args()

    cols = tuple(int(c) for c in args.cols.split(","))
    if len(cols) != 3:
        raise ValueError("--cols must contain exactly three integers")

    source, target = load_point_pairs(args.prefixes, cols=cols)
    transform, rmse = umeyama_alignment(source, target, allow_scaling=args.allow_scale)

    print("Estimated transform (source -> target):")
    print(transform)
    print(f"RMSE: {rmse:.4f}")

    result = {"transform": transform.tolist(), "rmse": rmse}
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
