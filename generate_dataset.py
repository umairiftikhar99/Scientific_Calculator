#!/usr/bin/env python3
"""Generate a Likert-scale dataset that approximates the thesis summary tables."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
N_RESPONDENTS = 225
CONSTRUCTS = {
    "AI": {"items": 5, "mean": 3.79, "sd": 0.65, "alpha": 0.84},
    "TC": {"items": 5, "mean": 3.58, "sd": 0.72, "alpha": 0.88},
    "PC": {"items": 5, "mean": 3.19, "sd": 0.69, "alpha": 0.81},
    "PS": {"items": 6, "mean": 3.89, "sd": 0.71, "alpha": 0.91},
}
CONSTRUCT_ORDER = ["AI", "TC", "PC", "PS"]

TARGET_CORR = np.array(
    [
        [1.00, 0.83, -0.12, 0.34],
        [0.83, 1.00, -0.15, 0.42],
        [-0.12, -0.15, 1.00, -0.25],
        [0.34, 0.42, -0.25, 1.00],
    ]
)
TARGET_REG_B = np.array([0.21, 0.36, -0.18])
TARGET_REG_R2 = 0.239
TARGET_MEDIATION = {
    "ai_tc": 1.03,
    "tc_ps": 0.36,
    "ai_direct": 0.09,
    "ai_total": 0.70,
}
TARGET_MODERATION = np.array([0.22, -0.17, 0.52])  # B coefficients for AI, PC, AIxPC

BASE_RNG = np.random.default_rng(seed=20250101)
BASE_LATENTS = BASE_RNG.standard_normal((N_RESPONDENTS, 3))
BASE_PS_NOISE = BASE_RNG.standard_normal(N_RESPONDENTS)
MEAS_NOISE = {
    name: BASE_RNG.standard_normal((N_RESPONDENTS, cfg["items"]))
    for name, cfg in CONSTRUCTS.items()
}

# Default parameters (tuned below).
COV_ATP_DEFAULT = np.array(
    [
        [1.00, 0.94678524, -0.11894836],
        [0.94678524, 1.00, -0.23459700],
        [-0.11894836, -0.23459700, 1.00],
    ]
)
PS_WEIGHTS_DEFAULT = np.array([-0.10329981, 0.73777611, -0.18582237])
PS_RESIDUAL_DEFAULT = 1.1455401194523032

# ---------------------------
# Helpers
# ---------------------------

def cronbach_alpha(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    k = values.shape[1]
    item_vars = values.var(axis=0, ddof=1)
    total_var = values.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


@dataclass
class RegressionResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r_squared: float
    adj_r_squared: float
    df_resid: int


def regression(y: np.ndarray, X: np.ndarray) -> RegressionResult:
    n = len(y)
    X_design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    fitted = X_design @ beta
    residuals = y - fitted
    df_resid = n - X_design.shape[1]
    sigma2 = (residuals @ residuals) / df_resid
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    nd = NormalDist()
    t_vals = beta / se
    p_vals = np.array([2 * (1 - nd.cdf(abs(t))) for t in t_vals])
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_resid
    return RegressionResult(beta=beta, se=se, t=t_vals, p=p_vals, r_squared=r_squared, adj_r_squared=adj_r_squared, df_resid=df_resid)


def standardized_betas(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y_std = y.std(ddof=1)
    stds = X.std(axis=0, ddof=1)
    return beta[1:] * stds / y_std


def compute_loading(alpha: float, items: int) -> float:
    r_bar = alpha / (items - alpha * (items - 1))
    return math.sqrt(r_bar / (1 - r_bar))


def find_transform(name: str, signal: np.ndarray, noise: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    cfg = CONSTRUCTS[name]
    target_mean = cfg["mean"]
    target_sd = cfg["sd"]
    target_alpha = cfg["alpha"]

    sigma_values = np.linspace(0.45, 0.95, 11)
    shift_values = np.linspace(target_mean - 0.4, target_mean + 0.4, 17)
    scale_values = np.linspace(0.45, 0.85, 13)

    best = None
    best_array = None
    best_params = None
    for sigma in sigma_values:
        adjusted = signal + sigma * noise
        for shift in shift_values:
            for scale in scale_values:
                likert = np.clip(np.round(shift + scale * adjusted), 1, 5).astype(int)
                composite = likert.mean(axis=1)
                mean_val = composite.mean()
                sd_val = composite.std(ddof=1)
                alpha_val = cronbach_alpha(likert)
                err = (
                    ((mean_val - target_mean) / 0.03) ** 2
                    + ((sd_val - target_sd) / 0.03) ** 2
                    + ((alpha_val - target_alpha) / 0.02) ** 2
                )
                if best is None or err < best:
                    best = err
                    best_array = likert
                    best_params = (sigma, shift, scale)
    assert best_array is not None and best_params is not None
    return best_array, best_params


# ---------------------------
# Generation logic
# ---------------------------

def sample_latents(cov_atp: np.ndarray, ps_weights: np.ndarray, ps_resid: float) -> Dict[str, np.ndarray]:
    chol = np.linalg.cholesky(cov_atp)
    atp = BASE_LATENTS @ chol.T
    ai_latent, tc_latent, pc_latent = atp.T
    ps_latent = ps_weights[0] * ai_latent + ps_weights[1] * tc_latent + ps_weights[2] * pc_latent + ps_resid * BASE_PS_NOISE
    ps_latent = (ps_latent - ps_latent.mean()) / ps_latent.std(ddof=1)
    return {
        "AI": ai_latent,
        "TC": tc_latent,
        "PC": pc_latent,
        "PS": ps_latent,
    }


def build_items(latents: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float, float]]]:
    signals: Dict[str, np.ndarray] = {}
    for name in CONSTRUCT_ORDER:
        cfg = CONSTRUCTS[name]
        loading = compute_loading(cfg["alpha"], cfg["items"])
        signals[name] = loading * latents[name][:, None]

    item_responses: Dict[str, np.ndarray] = {}
    params: Dict[str, Tuple[float, float, float]] = {}
    for name in CONSTRUCT_ORDER:
        likert, best_params = find_transform(name, signals[name], MEAS_NOISE[name])
        item_responses[name] = likert
        params[name] = best_params

    records = {}
    for name, arr in item_responses.items():
        for j in range(arr.shape[1]):
            records[f"{name}_{j+1}"] = arr[:, j]
        records[f"{name}_mean"] = arr.mean(axis=1)

    return pd.DataFrame(records), params


def summarize_metrics(df: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    composites = df[[f"{name}_mean" for name in CONSTRUCT_ORDER]]
    metrics["descriptives"] = {
        name: (composites[f"{name}_mean"].mean(), composites[f"{name}_mean"].std(ddof=1))
        for name in CONSTRUCT_ORDER
    }
    metrics["reliability"] = {
        name: cronbach_alpha(
            df[[f"{name}_{j+1}" for j in range(CONSTRUCTS[name]["items"])]].to_numpy()
        )
        for name in CONSTRUCT_ORDER
    }
    metrics["correlation"] = composites.corr()

    y_ps = composites["PS_mean"].to_numpy()
    X = composites[["AI_mean", "TC_mean", "PC_mean"]].to_numpy()
    reg = regression(y_ps, X)
    metrics["regression"] = {
        "beta": reg.beta[1:],
        "se": reg.se[1:],
        "beta_std": standardized_betas(y_ps, X, reg.beta),
        "r2": reg.r_squared,
        "adj_r2": reg.adj_r_squared,
    }

    X_ai = composites[["AI_mean"]].to_numpy()
    reg_ai_tc = regression(composites["TC_mean"].to_numpy(), X_ai)
    reg_tc_ps = regression(y_ps, composites[["TC_mean", "AI_mean"]].to_numpy())
    reg_ai_ps_total = regression(y_ps, X_ai)
    metrics["mediation"] = {
        "ai_tc": reg_ai_tc.beta[1],
        "tc_ps": reg_tc_ps.beta[1],
        "ai_direct": reg_tc_ps.beta[2],
        "ai_total": reg_ai_ps_total.beta[1],
    }

    ai = composites[["AI_mean"]].to_numpy()
    pc = composites[["PC_mean"]].to_numpy()
    interaction = ai * pc
    reg_mod = regression(y_ps, np.column_stack([ai, pc, interaction]))
    metrics["moderation"] = reg_mod.beta[1:]
    return metrics


def score_metrics(metrics: Dict[str, object]) -> float:
    corr = metrics["correlation"].to_numpy()
    corr_error = np.linalg.norm((corr - TARGET_CORR)[np.triu_indices(4, 1)]) / 4

    reg_beta = metrics["regression"]["beta"]
    reg_error = np.linalg.norm((reg_beta - TARGET_REG_B) / np.array([0.05, 0.05, 0.05]))
    r2_error = abs(metrics["regression"]["r2"] - TARGET_REG_R2) / 0.02

    med = metrics["mediation"]
    med_error = sum(
        abs(med[key] - TARGET_MEDIATION[key]) / 0.1 for key in TARGET_MEDIATION
    )

    mod_error = np.linalg.norm((metrics["moderation"] - TARGET_MODERATION) / np.array([0.1, 0.1, 0.1]))

    return corr_error + reg_error + r2_error + med_error + mod_error


def generate_dataset(
    cov_atp: np.ndarray = COV_ATP_DEFAULT,
    ps_weights: np.ndarray = PS_WEIGHTS_DEFAULT,
    ps_residual: float = PS_RESIDUAL_DEFAULT,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float, float]], Dict[str, object]]:
    latents = sample_latents(cov_atp, ps_weights, ps_residual)
    df, params = build_items(latents)
    metrics = summarize_metrics(df)
    return df, params, metrics


# ---------------------------
# Script entry point
# ---------------------------
if __name__ == "__main__":
    df, params, metrics = generate_dataset()
    output_path = Path("data/thesis_synthetic_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved dataset to", output_path)
    print()

    print("Table 4.1 – Descriptive statistics")
    for name in CONSTRUCT_ORDER:
        mean_val, sd_val = metrics["descriptives"][name]
        target = CONSTRUCTS[name]
        print(f"{name:>2}: mean={mean_val:.3f} sd={sd_val:.3f} (target mean {target['mean']}, sd {target['sd']})")
    print()

    print("Table 4.2 – Reliability")
    for name in CONSTRUCT_ORDER:
        print(f"{name:>2}: alpha={metrics['reliability'][name]:.3f} target={CONSTRUCTS[name]['alpha']} params={params[name]}")
    print()

    print("Table 4.3 – Correlations")
    print(metrics["correlation"].round(3))
    print()

    reg = metrics["regression"]
    print("Table 4.4 – Multiple regression")
    for name, b, se, beta_std in zip(["AI", "TC", "PC"], reg["beta"], reg["se"], reg["beta_std"]):
        print(f"{name}: B={b:.2f} SE={se:.2f} Beta={beta_std:.2f}")
    print(f"Model: R^2={reg['r2']:.3f} Adj R^2={reg['adj_r2']:.3f}")
    print()

    print("Table 4.5 – Mediation")
    med = metrics["mediation"]
    print(f"AI -> TC: B={med['ai_tc']:.2f}")
    print(f"TC -> PS (with AI): B={med['tc_ps']:.2f}")
    print(f"AI -> PS direct: B={med['ai_direct']:.2f}")
    print(f"AI -> PS total: B={med['ai_total']:.2f}")
    print()

    print("Table 4.6 – Moderation")
    for label, value in zip(["AI", "PC", "AIxPC"], metrics["moderation"]):
        print(f"{label}: B={value:.2f}")
    print()

    print("Overall error score:", score_metrics(metrics))
