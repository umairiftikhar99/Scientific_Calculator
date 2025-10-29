"""Generate a synthetic dataset matching the supplied thesis tables.

The generator works in three stages:

1. Construct the scale means (AI, TC, PC, PS) so that the sample means,
   standard deviations, and Pearson correlations match the published values
   exactly.
2. Expand each construct into tau-equivalent item responses that reproduce the
   requested Cronbach's alpha coefficients while keeping the person-level scale
   scores unchanged.
3. Save the full item-level data along with the derived variables needed for the
   moderation analysis.

The default output is a CSV file named ``generated_dataset.csv`` in this
directory.  Run the module as a script to regenerate the file and to view a
short summary of the achieved statistics.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ---- Target specification -------------------------------------------------

N_OBS = 225
CONSTRUCT_ORDER = ["AI", "TC", "PC", "PS"]
TARGET_MEANS = np.array([3.79, 3.58, 3.19, 3.89])
TARGET_SDS = np.array([0.65, 0.72, 0.69, 0.71])
TARGET_CORR = np.array(
    [
        [1.00, 0.83, -0.12, 0.34],
        [0.83, 1.00, -0.15, 0.42],
        [-0.12, -0.15, 1.00, -0.25],
        [0.34, 0.42, -0.25, 1.00],
    ]
)


@dataclass(frozen=True)
class ConstructSpec:
    """Container for the item-level requirements of a construct."""

    name: str
    n_items: int
    alpha: float
    column: int  # column index in the scale score matrix


CONSTRUCT_SPECS: Tuple[ConstructSpec, ...] = (
    ConstructSpec("AI", 5, 0.84, 0),
    ConstructSpec("TC", 5, 0.88, 1),
    ConstructSpec("PC", 5, 0.81, 2),
    ConstructSpec("PS", 6, 0.91, 3),
)

TARGET_COV = TARGET_SDS[:, None] * TARGET_SDS[None, :] * TARGET_CORR
OUTPUT_PATH = Path(__file__).with_name("generated_dataset.csv")


# ---- Linear algebra helpers ----------------------------------------------

def _orthonormal_basis(n_rows: int, n_cols: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``n_rows × n_cols`` matrix with orthonormal columns.

    Each column is mean-centered to guarantee that the final sample means of the
    generated data are controlled solely by the shifts applied later.
    """

    basis = np.empty((n_rows, n_cols), dtype=float)
    for j in range(n_cols):
        vec = rng.standard_normal(n_rows)
        vec -= vec.mean()  # enforce zero column sum
        for k in range(j):
            vec -= np.dot(basis[:, k], vec) * basis[:, k]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            raise RuntimeError("Failed to construct an orthonormal column.")
        basis[:, j] = vec / norm
    return basis


def _generate_scale_scores(rng: np.random.Generator) -> np.ndarray:
    """Create the 4 scale scores with exact means, SDs, and correlations."""

    basis = _orthonormal_basis(N_OBS, len(CONSTRUCT_ORDER), rng)
    eigvals, eigvecs = np.linalg.eigh(TARGET_COV)
    # Guard against tiny negative eigenvalues caused by rounding.
    eigvals = np.maximum(eigvals, 0.0)
    transform = basis @ (np.sqrt(eigvals)[:, None] * eigvecs.T)
    centered = np.sqrt(N_OBS - 1) * transform
    return centered + TARGET_MEANS


# ---- Item generation ------------------------------------------------------

def _cronbach_alpha(items: np.ndarray) -> float:
    """Compute Cronbach's alpha for the provided item matrix."""

    k = items.shape[1]
    covariance = np.cov(items, rowvar=False, ddof=1)
    total_variance = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1.0)) * (1.0 - np.trace(covariance) / total_variance)


def _generate_items(
    scale_scores: np.ndarray, specs: Iterable[ConstructSpec], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Expand the scale scores into item-level responses.

    The construction keeps the per-person scale means untouched: the noises added
    to each item sum to zero across the items belonging to the same construct.
    The variance of the noise is calibrated so that the resulting Cronbach's
    alpha matches the target value for each construct under the tau-equivalent
    assumption.
    """

    items: Dict[str, np.ndarray] = {}
    for spec in specs:
        scale = scale_scores[:, spec.column]
        variance_scale = scale.var(ddof=1)
        tau2 = spec.n_items * variance_scale * (1.0 - spec.alpha)
        noise = rng.normal(0.0, np.sqrt(tau2), size=(N_OBS, spec.n_items))
        noise -= noise.mean(axis=1, keepdims=True)
        items[spec.name] = scale[:, None] + noise
    return items


# ---- Diagnostics ----------------------------------------------------------

def _scale_matrix_from_items(items: Dict[str, np.ndarray]) -> np.ndarray:
    """Return the scale means computed from the item matrices."""

    return np.column_stack([items[name].mean(axis=1) for name in CONSTRUCT_ORDER])


def _regression_summary(scale_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run PS ~ AI + TC + PC and return (coefficients, standard errors, R²)."""

    y = scale_matrix[:, 3]
    X = np.column_stack([np.ones(N_OBS), scale_matrix[:, :3]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    ss_res = np.sum(residuals**2)
    df_resid = N_OBS - X.shape[1]
    mse = ss_res / df_resid
    cov_beta = mse * np.linalg.inv(X.T @ X)
    standard_errors = np.sqrt(np.diag(cov_beta))
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return beta, standard_errors, r_squared


def _print_diagnostics(scale_matrix: np.ndarray, items: Dict[str, np.ndarray]) -> None:
    """Display a concise comparison between achieved and target metrics."""

    means = scale_matrix.mean(axis=0)
    sds = scale_matrix.std(axis=0, ddof=1)
    corr = np.corrcoef(scale_matrix, rowvar=False)

    print("Sample size:", scale_matrix.shape[0])
    print("\nScale descriptives (mean, sd):")
    for name, mean, sd, t_mean, t_sd in zip(CONSTRUCT_ORDER, means, sds, TARGET_MEANS, TARGET_SDS):
        print(f"  {name}: {mean:.5f} (target {t_mean:.2f}), sd {sd:.5f} (target {t_sd:.2f})")

    print("\nScale-level correlations:")
    for i, name_i in enumerate(CONSTRUCT_ORDER):
        row = [f"{name_i}-{CONSTRUCT_ORDER[j]}: {corr[i, j]:.5f}" for j in range(len(CONSTRUCT_ORDER))]
        print("  " + ", ".join(row))

    print("\nCronbach's alpha (target in brackets):")
    for spec in CONSTRUCT_SPECS:
        alpha = _cronbach_alpha(items[spec.name])
        print(f"  {spec.name}: {alpha:.5f} (target {spec.alpha:.2f})")

    beta, se, r_squared = _regression_summary(scale_matrix)
    coef_names = ["Intercept"] + CONSTRUCT_ORDER[:-1]
    print("\nRegression PS ~ AI + TC + PC:")
    for name, coef, err in zip(coef_names, beta, se):
        print(f"  {name}: B = {coef:.5f}, SE = {err:.5f}")
    print(f"  R^2 = {r_squared:.5f}")


# ---- CSV export -----------------------------------------------------------

def _build_output_matrix(
    items: Dict[str, np.ndarray], scale_matrix: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Assemble the final dataset matrix and its header."""

    columns: List[str] = []
    matrices: List[np.ndarray] = []
    for spec in CONSTRUCT_SPECS:
        matrices.append(items[spec.name])
        columns.extend(f"{spec.name}{i + 1}" for i in range(spec.n_items))

    matrices.append(scale_matrix)
    columns.extend(CONSTRUCT_ORDER)

    ai_centered = scale_matrix[:, 0] - TARGET_MEANS[0]
    pc_centered = scale_matrix[:, 2] - TARGET_MEANS[2]
    interaction = ai_centered * pc_centered
    matrices.extend([ai_centered[:, None], pc_centered[:, None], interaction[:, None]])
    columns.extend(["AI_c", "PC_c", "AIxPC"])

    return columns, np.column_stack(matrices)


def _write_csv(path: Path, header: Iterable[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in data:
            writer.writerow(f"{value:.10f}" for value in row)


# ---- Public API -----------------------------------------------------------

def generate_dataset(seed: int = 2024) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return the scale scores and item matrices for the synthetic dataset."""

    rng = np.random.default_rng(seed)
    scale_scores = _generate_scale_scores(rng)
    items = _generate_items(scale_scores, CONSTRUCT_SPECS, rng)
    return scale_scores, items


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=2024, help="Random seed used for reproducibility (default: 2024)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the CSV file to generate (default: analysis/generated_dataset.csv).",
    )
    args = parser.parse_args(argv)

    scale_scores, items = generate_dataset(seed=args.seed)
    scale_from_items = _scale_matrix_from_items(items)

    _print_diagnostics(scale_from_items, items)

    header, matrix = _build_output_matrix(items, scale_from_items)
    _write_csv(args.output, header, matrix)
    print(f"\nSaved dataset to {args.output}")


if __name__ == "__main__":
    main()
