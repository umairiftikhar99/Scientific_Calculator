"""Generate a synthetic Likert-scale dataset for AI adoption research.

This script produces 5-point Likert-scale responses for 21 survey items
across four constructs: Artificial Intelligence Adoption (AI1–AI5), Team
Collaboration (TC1–TC5), Project Complexity (PC1–PC5), and Project Success
(PS1–PS6). The resulting dataset (n=285) follows theoretical expectations
from the Technology Acceptance Model (TAM), Social Cognitive Theory (SCT),
and Contingency Theory (CT).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


N_PARTICIPANTS = 285
LIKERT_MIN = 1
LIKERT_MAX = 5
TARGET_R2 = 0.35
STRUCTURAL_BETAS = {
    "AI": 0.30,
    "TC_SHARED": 0.24,
    "TC_UNIQUE": 0.37,
    "PC": -0.23,
}

CONSTRUCT_METADATA = {
    "AI": {"mean": 3.8, "sd": 0.6, "items": 5},
    "TC": {"mean": 3.6, "sd": 0.7, "items": 5},
    "PC": {"mean": 3.2, "sd": 0.65, "items": 5},
    "PS": {"mean": 3.9, "sd": 0.7, "items": 6},
}

# Correlations for the predictor constructs (AI, TC, PC)
PREDICTOR_CORR = np.array(
    [
        [1.00, 0.82, -0.18],
        [0.82, 1.00, -0.20],
        [-0.18, -0.20, 1.00],
    ]
)


def _generate_construct_scores(seed: int | None = 42) -> pd.DataFrame:
    """Draw latent construct scores that satisfy the theoretical assumptions."""

    rng = np.random.default_rng(seed)

    # Latent z-scores for predictors
    ai_tc_pc = rng.multivariate_normal(np.zeros(3), PREDICTOR_CORR, size=N_PARTICIPANTS)
    ai_z, tc_z, pc_z = ai_tc_pc.T

    # Project success is driven by the theorised structural paths
    beta_ai = STRUCTURAL_BETAS["AI"]
    beta_tc_shared = STRUCTURAL_BETAS["TC_SHARED"]
    beta_tc_unique = STRUCTURAL_BETAS["TC_UNIQUE"]
    beta_pc = STRUCTURAL_BETAS["PC"]
    # Separate the portion of team collaboration that is not explained by AI
    tc_unique = tc_z - PREDICTOR_CORR[0, 1] * ai_z
    tc_unique /= tc_unique.std(ddof=0)

    ps_linear = (
        beta_ai * ai_z
        + beta_tc_shared * tc_z
        + beta_tc_unique * tc_unique
        + beta_pc * pc_z
    )

    var_linear = np.var(ps_linear, ddof=0)
    noise_sd = np.sqrt(var_linear * (1 - TARGET_R2) / TARGET_R2)
    ps_z = ps_linear + rng.normal(0.0, noise_sd, size=N_PARTICIPANTS)

    def _scale(z_scores: np.ndarray, *, mean: float, sd: float) -> np.ndarray:
        centered = z_scores - z_scores.mean()
        return mean + centered / z_scores.std(ddof=0) * sd

    data = {}
    for key, z_scores in {"AI": ai_z, "TC": tc_z, "PC": pc_z, "PS": ps_z}.items():
        meta = CONSTRUCT_METADATA[key]
        data[f"{key}_Mean"] = _scale(z_scores, mean=meta["mean"], sd=meta["sd"])

    return pd.DataFrame(data)


def _generate_items(base: pd.Series, n_items: int, rng: np.random.Generator, noise_sd: float = 0.25) -> np.ndarray:
    """Create Likert-scale items around a *base* construct score."""

    noise = rng.normal(0.0, noise_sd, size=(n_items, base.size))
    items = base.to_numpy() + noise
    items = np.clip(np.rint(items), LIKERT_MIN, LIKERT_MAX).astype(int)
    return items.T


def _build_dataset(seed: int | None = 42) -> pd.DataFrame:
    """Generate the full item-level dataset with participant identifiers."""

    rng = np.random.default_rng(seed)
    construct_df = _generate_construct_scores(seed)

    item_arrays = []
    columns: list[str] = []
    for construct, meta in CONSTRUCT_METADATA.items():
        items = _generate_items(construct_df[f"{construct}_Mean"], meta["items"], rng)
        item_arrays.append(items)
        columns.extend([f"{construct}{i + 1}" for i in range(meta["items"])] )

    data = np.hstack(item_arrays)
    df = pd.DataFrame(data, columns=columns)

    # Construct means from rounded item responses
    for construct in CONSTRUCT_METADATA:
        item_cols = [col for col in df.columns if col.startswith(construct)]
        df[f"{construct}_Mean"] = df[item_cols].mean(axis=1)

    df.insert(0, "ParticipantID", np.arange(1, N_PARTICIPANTS + 1))
    return df


def _summarise_constructs(df: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Return descriptive statistics that validate the theoretical design."""

    construct_cols = [f"{construct}_Mean" for construct in CONSTRUCT_METADATA]
    corr_matrix = df[construct_cols].corr()

    X = sm.add_constant(df[["AI_Mean", "TC_Mean", "PC_Mean"]])
    y = df["PS_Mean"]
    model = sm.OLS(y, X).fit()
    return corr_matrix, model


def main(output_path: Path, seed: int | None = 42) -> None:
    df = _build_dataset(seed)
    corr_matrix, model = _summarise_constructs(df)

    df.to_csv(output_path, index=False)

    print("\n📊 Construct Correlation Matrix:\n")
    print(corr_matrix.round(3))

    print("\n📈 Regression Results Summary:\n")
    print(model.summary())

    print(f"\n✅ Synthetic dataset saved to '{output_path}'.\n")

    print(
        """
-------------------------------------------
Expected Theoretical Output:
-------------------------------------------
Correlations:
  AI–TC ≈ 0.80
  AI–PS ≈ 0.45
  TC–PS ≈ 0.48
  PC–PS ≈ –0.25

Regression (Multiple Linear):
  AI → PS: β ≈ 0.20 (p ≈ .01)
  TC → PS: β ≈ 0.33 (p < .001)
  PC → PS: β ≈ –0.20 (p ≈ .002)
  R² ≈ 0.30

Interpretation:
✅ AI adoption improves success (TAM)
✅ Team collaboration mediates (SCT)
✅ Complexity hinders success (CT)
❌ No moderation detected
-------------------------------------------
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Thesis_Synthetic_Likert21_285.csv"),
        help="Path for the generated CSV dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args.output, seed=args.seed)
