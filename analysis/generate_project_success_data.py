"""Generate a synthetic dataset that matches the provided project success study statistics.

The script simulates latent constructs (AI enablement, team collaboration, project
complexity, and project success) and derives item-level responses that align with the
reported descriptive statistics, correlations, and reliabilities. The resulting dataset
is saved to ``analysis/data/project_success_synthetic.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstructSpec:
    """Specification for a construct measured by multiple Likert items."""

    mean: float
    sd: float
    items: int
    alpha: float


DEFAULT_N_RESPONDENTS = 225

CONSTRUCT_SPECS: Dict[str, ConstructSpec] = {
    "AI": ConstructSpec(mean=3.79, sd=0.65, items=5, alpha=0.84),
    "TeamCollaboration": ConstructSpec(mean=3.58, sd=0.72, items=5, alpha=0.88),
    "ProjectComplexity": ConstructSpec(mean=3.19, sd=0.69, items=5, alpha=0.81),
    "ProjectSuccess": ConstructSpec(mean=3.89, sd=0.71, items=6, alpha=0.91),
}

OBSERVED_CORRELATIONS = pd.DataFrame(
    data=[
        [1.0, 0.83, -0.12, 0.34],
        [0.83, 1.0, -0.15, 0.42],
        [-0.12, -0.15, 1.0, -0.25],
        [0.34, 0.42, -0.25, 1.0],
    ],
    index=CONSTRUCT_SPECS.keys(),
    columns=CONSTRUCT_SPECS.keys(),
)


def average_inter_item_correlation(alpha: float, items: int) -> float:
    """Return the average inter-item correlation implied by Cronbach's alpha."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if items < 2:
        raise ValueError("items must be at least 2 for reliability")
    return alpha / (items - alpha * (items - 1))


def latent_variance(target_sd: float, alpha: float, items: int) -> float:
    """Compute latent variance required for the average scale to match the SD."""

    r_bar = average_inter_item_correlation(alpha, items)
    numerator = target_sd**2
    denominator = 1 + (1 - r_bar) / (r_bar * items)
    return numerator / denominator


def build_latent_covariance(specs: Dict[str, ConstructSpec]) -> pd.DataFrame:
    """Create the covariance matrix for the latent construct scores."""

    std_latents = {
        name: np.sqrt(latent_variance(spec.sd, spec.alpha, spec.items))
        for name, spec in specs.items()
    }
    latent_correlations = np.eye(len(specs))
    names = list(specs.keys())
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            name_j = names[j]
            observed = OBSERVED_CORRELATIONS.loc[name_i, name_j]
            reliability_i = specs[name_i].alpha
            reliability_j = specs[name_j].alpha
            attenuation = np.sqrt(reliability_i * reliability_j)
            latent_value = observed / attenuation if attenuation > 0 else observed
            latent_correlations[i, j] = latent_correlations[j, i] = latent_value
    scale_matrix = np.diag([std_latents[name] for name in names])
    covariance = scale_matrix @ latent_correlations @ scale_matrix
    return pd.DataFrame(covariance, index=specs.keys(), columns=specs.keys())


TARGET_MEANS = pd.Series({name: spec.mean for name, spec in CONSTRUCT_SPECS.items()})
TARGET_SDS = pd.Series({name: spec.sd for name, spec in CONSTRUCT_SPECS.items()})
TARGET_CORRELATIONS = OBSERVED_CORRELATIONS


def _quantize_likert(values: np.ndarray, target_mean: float) -> np.ndarray:
    """Map continuous item responses onto a discrete 1-5 Likert scale."""

    shift = 0.0
    for _ in range(50):
        rounded = np.clip(np.rint(values + shift), 1, 5)
        diff = rounded.mean() - target_mean
        if abs(diff) <= 0.01:
            return rounded.astype(int)
        shift -= diff * 0.5

    rounded = np.clip(np.rint(values + shift), 1, 5)
    return rounded.astype(int)


def generate_items(
    latent_scores: np.ndarray, spec: ConstructSpec, *, rng: np.random.Generator
) -> np.ndarray:
    """Generate item-level responses from latent scores."""

    r_bar = average_inter_item_correlation(spec.alpha, spec.items)
    latent_var = latent_variance(spec.sd, spec.alpha, spec.items)
    noise_var = latent_var * (1 - r_bar) / r_bar
    noise = rng.normal(0.0, np.sqrt(noise_var), size=(latent_scores.shape[0], spec.items))
    items = latent_scores[:, None] + noise
    items -= items.mean() - spec.mean
    quantized = _quantize_likert(items, spec.mean)
    return quantized


def _simulate_latent_scores(
    n: int, *, seed: int, specs: Dict[str, ConstructSpec]
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    covariance = build_latent_covariance(specs)
    means = np.array([spec.mean for spec in specs.values()])
    latents = rng.multivariate_normal(mean=means, cov=covariance, size=n)
    return latents


def _generate_dataset(
    n: int,
    *,
    seed: int,
    specs: Dict[str, ConstructSpec],
) -> pd.DataFrame:
    latents = _simulate_latent_scores(n, seed=seed, specs=specs)
    data: Dict[str, List[float]] = {}
    for idx, (name, spec) in enumerate(specs.items()):
        rng = np.random.default_rng(seed + idx + 1)
        items = generate_items(latents[:, idx], spec, rng=rng)
        for item_idx in range(spec.items):
            column = f"{name}_Item{item_idx + 1}"
            data[column] = items[:, item_idx]
        data[f"{name}_Score"] = items.mean(axis=1)

    df = pd.DataFrame(data)
    df.insert(0, "RespondentID", np.arange(1, n + 1))
    return df


def _summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = [f"{name}_Score" for name in CONSTRUCT_SPECS]
    summary = df[score_columns].agg(["mean", "std"]).T
    summary.index = [name.replace("_Score", "") for name in summary.index]
    correlations = df[score_columns].corr()
    correlations.index = [name.replace("_Score", "") for name in correlations.index]
    correlations.columns = [name.replace("_Score", "") for name in correlations.columns]
    return summary, correlations


def _evaluation_error(summary: pd.DataFrame, correlations: pd.DataFrame) -> float:
    mean_diff = (summary["mean"] - TARGET_MEANS).abs().sum()
    sd_diff = (summary["std"] - TARGET_SDS).abs().sum()
    corr_diff = (correlations - TARGET_CORRELATIONS).abs().values.sum() / 2
    return float(mean_diff + sd_diff + corr_diff)


def search_best_dataset(
    *,
    candidate_ns: List[int],
    attempts_per_n: int,
    tolerance: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float, float]:
    base_rng = np.random.default_rng(20240220)
    best_result: Optional[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float]
    ] = None
    best_error = float("inf")

    for n in candidate_ns:
        for _ in range(attempts_per_n):
            seed = int(base_rng.integers(0, 2**32 - 1))
            df = _generate_dataset(n, seed=seed, specs=CONSTRUCT_SPECS)
            summary, correlations = _summarize(df)
            error = _evaluation_error(summary, correlations)
            if error < best_error:
                best_error = error
                best_result = (df, summary, correlations, n, seed)
            if best_error <= tolerance:
                break
        if best_error <= tolerance:
            break

    if best_result is None:
        raise RuntimeError("Failed to generate dataset")

    df, summary, correlations, n, seed = best_result
    return df, summary, correlations, n, seed, best_error


def main() -> None:
    candidate_ns = list(range(DEFAULT_N_RESPONDENTS - 20, DEFAULT_N_RESPONDENTS + 21, 5))
    df, summary, correlations, n, seed, error = search_best_dataset(
        candidate_ns=candidate_ns,
        attempts_per_n=250,
        tolerance=0.25,
    )

    output_dir = Path("analysis/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "project_success_synthetic.csv"
    df.to_csv(output_path, index=False)

    print(f"Synthetic data saved to {output_path} (n={n}, seed={seed})")
    print("\nScale means and standard deviations:\n", summary)
    print("\nScale correlations:\n", correlations)
    print(f"\nTotal deviation score: {error:.3f}")


if __name__ == "__main__":
    main()
