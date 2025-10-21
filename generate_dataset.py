import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)

N_RESPONDENTS = 225
CONSTRUCT_ORDER = ["AI", "TC", "PC", "PS"]
ITEM_COUNTS = {"AI": 5, "TC": 5, "PC": 5, "PS": 6}

TARGET_MEANS = {"AI": 3.79, "TC": 3.58, "PC": 3.19, "PS": 3.89}
TARGET_SDS = {"AI": 0.65, "TC": 0.72, "PC": 0.69, "PS": 0.71}
TARGET_ALPHA = {"AI": 0.84, "TC": 0.88, "PC": 0.81, "PS": 0.91}
TARGET_CORR = {
    ("AI", "TC"): 0.83,
    ("AI", "PC"): -0.12,
    ("AI", "PS"): 0.34,
    ("TC", "PC"): -0.15,
    ("TC", "PS"): 0.42,
    ("PC", "PS"): -0.25,
}
TARGET_REGRESSION = {
    "intercept": 2.11,
    "AI": 0.21,
    "TC": 0.36,
    "PC": -0.18,
    "r2": 0.239,
}
TARGET_MEDIATION = {
    "a": 1.03,  # AI -> TC
    "b": 0.36,  # TC -> PS (controlling AI)
    "c_prime": 0.09,  # AI direct -> PS
    "total": 0.70,  # AI total effect on PS
}
TARGET_MODERATION = {
    "AI": 0.22,
    "PC": -0.17,
    "interaction": 0.52,
}

BASE_PARAMS = {
    "r_at": 0.97,
    "r_ap": 0.52,
    "r_tp": 0.57,
    "r_ac": -0.12,
    "r_tc": -0.16,
    "r_pc": -0.28,
    "err_AI": 0.60,
    "err_TC": 0.75,
    "err_PC": 0.90,
    "err_PS": 0.70,
}

DEFAULT_PARAMS = {
    "r_at": 0.9063,
    "r_ap": 0.4873,
    "r_tp": 0.5445,
    "r_ac": -0.0505,
    "r_tc": -0.0644,
    "r_pc": -0.1078,
    "err_AI": 0.8933,
    "err_TC": 0.7381,
    "err_PC": 0.7975,
    "err_PS": 0.5845,
}


def average_inter_item_corr(alpha: float, k: int) -> float:
    return alpha / (k - (k - 1) * alpha)


def cronbach_alpha(df: pd.DataFrame) -> float:
    values = df.to_numpy(dtype=float)
    k = values.shape[1]
    item_variances = values.var(axis=0, ddof=1)
    total_var = values.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0.0
    return (k / (k - 1)) * (1 - item_variances.sum() / total_var)


def build_corr_matrix(params: dict) -> np.ndarray:
    mat = np.array([
        [1.0, params["r_at"], params["r_ac"], params["r_ap"]],
        [params["r_at"], 1.0, params["r_tc"], params["r_tp"]],
        [params["r_ac"], params["r_tc"], 1.0, params["r_pc"]],
        [params["r_ap"], params["r_tp"], params["r_pc"], 1.0],
    ])
    min_eig = np.linalg.eigvalsh(mat).min()
    if min_eig < 1e-6:
        raise ValueError("Correlation matrix not positive definite")
    return mat


def calibrate_items(raw_items: np.ndarray, target_mean: float, target_sd: float) -> tuple[np.ndarray, np.ndarray]:
    scale = target_sd / (raw_items.mean(axis=1).std(ddof=1) + 1e-8)
    shift = target_mean - raw_items.mean()
    for _ in range(30):
        transformed = np.clip(np.round(raw_items * scale + shift), 1, 5)
        composite = transformed.mean(axis=1)
        mean_diff = target_mean - composite.mean()
        sd = composite.std(ddof=1)
        sd_ratio = target_sd / (sd + 1e-8)
        shift += 0.5 * mean_diff
        scale *= sd_ratio ** 0.5
        if abs(mean_diff) < 0.002 and abs(target_sd - sd) < 0.002:
            break
    best = None
    for s in np.linspace(scale * 0.97, scale * 1.03, 7):
        for sh in np.linspace(shift - 0.08, shift + 0.08, 7):
            transformed = np.clip(np.round(raw_items * s + sh), 1, 5)
            composite = transformed.mean(axis=1)
            score = (composite.mean() - target_mean) ** 2 + (composite.std(ddof=1) - target_sd) ** 2
            if best is None or score < best[0]:
                best = (score, transformed, composite)
    return best[1], best[2]


def generate_dataset(params: dict, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corr = build_corr_matrix(params)
    latent = rng.multivariate_normal(np.zeros(len(CONSTRUCT_ORDER)), corr, size=N_RESPONDENTS)

    data = {}
    for idx, construct in enumerate(CONSTRUCT_ORDER):
        k = ITEM_COUNTS[construct]
        base_r = average_inter_item_corr(TARGET_ALPHA[construct], k)
        var_err = (1 / base_r) - 1
        var_err *= params[f"err_{construct}"]
        latent_scores = latent[:, idx]
        noise = rng.normal(0, math.sqrt(var_err), size=(N_RESPONDENTS, k))
        raw_items = latent_scores[:, None] + noise
        transformed, composite = calibrate_items(raw_items, TARGET_MEANS[construct], TARGET_SDS[construct])
        for j in range(k):
            data[f"{construct}{j+1}"] = transformed[:, j].astype(int)
        data[f"{construct}_mean"] = composite
    df = pd.DataFrame(data)
    ordered_cols = [c for c in df.columns if not c.endswith("_mean")] + [c for c in df.columns if c.endswith("_mean")]
    return df[ordered_cols]


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {"means": {}, "sds": {}, "alpha": {}, "corr": {}, "regression": {}, "mediation": {}, "moderation": {}}
    for construct in CONSTRUCT_ORDER:
        cols = [c for c in df.columns if c.startswith(construct) and not c.endswith("_mean")]
        composite = df[f"{construct}_mean"]
        metrics["means"][construct] = composite.mean()
        metrics["sds"][construct] = composite.std(ddof=1)
        metrics["alpha"][construct] = cronbach_alpha(df[cols])

    composites = df[[f"{c}_mean" for c in CONSTRUCT_ORDER]]
    corr_matrix = composites.corr()
    for (a, b), target in TARGET_CORR.items():
        metrics["corr"][(a, b)] = corr_matrix.loc[f"{a}_mean", f"{b}_mean"]

    X = sm.add_constant(composites[["AI_mean", "TC_mean", "PC_mean"]])
    y = composites["PS_mean"]
    model = sm.OLS(y, X).fit()
    metrics["regression"] = {
        "intercept": model.params.iloc[0],
        "AI": model.params.loc["AI_mean"],
        "TC": model.params.loc["TC_mean"],
        "PC": model.params.loc["PC_mean"],
        "r2": model.rsquared,
    }

    med_a = sm.OLS(composites["TC_mean"], sm.add_constant(composites["AI_mean"])).fit()
    med_b = sm.OLS(composites["PS_mean"], sm.add_constant(composites[["AI_mean", "TC_mean"]])).fit()
    med_total = sm.OLS(composites["PS_mean"], sm.add_constant(composites["AI_mean"])).fit()
    metrics["mediation"] = {
        "a": med_a.params.loc["AI_mean"],
        "b": med_b.params.loc["TC_mean"],
        "c_prime": med_b.params.loc["AI_mean"],
        "total": med_total.params.loc["AI_mean"],
    }

    interaction = composites["AI_mean"] * composites["PC_mean"]
    mod_X = sm.add_constant(pd.DataFrame({
        "AI": composites["AI_mean"],
        "PC": composites["PC_mean"],
        "interaction": interaction,
    }))
    mod_model = sm.OLS(composites["PS_mean"], mod_X).fit()
    metrics["moderation"] = {
        "AI": mod_model.params.loc["AI"],
        "PC": mod_model.params.loc["PC"],
        "interaction": mod_model.params.loc["interaction"],
    }
    return metrics


def metric_error(metrics: dict) -> float:
    def add(score, observed, target, tolerance, weight=1.0):
        return score + weight * ((observed - target) / tolerance) ** 2

    score = 0.0
    for construct in CONSTRUCT_ORDER:
        score = add(score, metrics["means"][construct], TARGET_MEANS[construct], 0.01, weight=3.5)
        score = add(score, metrics["sds"][construct], TARGET_SDS[construct], 0.015, weight=3.5)
        score = add(score, metrics["alpha"][construct], TARGET_ALPHA[construct], 0.015, weight=3.0)

    for key, target in TARGET_CORR.items():
        score = add(score, metrics["corr"][key], target, 0.02, weight=2.5)

    for key, target in TARGET_REGRESSION.items():
        tol = 0.05 if key == "intercept" else 0.025
        weight = 3.0 if key != "r2" else 2.0
        score = add(score, metrics["regression"][key], target, tol, weight)

    for key, target in TARGET_MEDIATION.items():
        score = add(score, metrics["mediation"][key], target, 0.04, weight=2.0)

    score = add(score, metrics["moderation"]["AI"], TARGET_MODERATION["AI"], 0.05, weight=1.5)
    score = add(score, metrics["moderation"]["PC"], TARGET_MODERATION["PC"], 0.05, weight=1.5)
    score = add(score, metrics["moderation"]["interaction"], TARGET_MODERATION["interaction"], 0.06, weight=1.2)
    return score


PARAM_KEYS = [
    "r_at",
    "r_ap",
    "r_tp",
    "r_ac",
    "r_tc",
    "r_pc",
    "err_AI",
    "err_TC",
    "err_PC",
    "err_PS",
]

PARAM_BOUNDS = [
    (0.75, 0.99),  # r_at
    (0.25, 0.65),  # r_ap
    (0.30, 0.70),  # r_tp
    (-0.30, -0.02),  # r_ac
    (-0.30, -0.02),  # r_tc
    (-0.40, -0.10),  # r_pc
    (0.45, 0.95),  # err_AI
    (0.45, 0.95),  # err_TC
    (0.45, 0.95),  # err_PC
    (0.45, 0.95),  # err_PS
]


def vector_to_params(vector: np.ndarray) -> dict:
    return {key: float(value) for key, value in zip(PARAM_KEYS, vector)}


def optimize_parameters(maxiter: int = 40, seed: int = 21) -> tuple[dict, pd.DataFrame, dict]:
    from scipy.optimize import differential_evolution

    def objective(vector: np.ndarray) -> float:
        params = vector_to_params(vector)
        try:
            df = generate_dataset(params, seed=seed)
            metrics = compute_metrics(df)
        except ValueError:
            return 1e9
        return metric_error(metrics)

    result = differential_evolution(
        objective,
        PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=12,
        tol=0.02,
        mutation=(0.5, 0.9),
        recombination=0.7,
        seed=seed,
        polish=True,
        updating="immediate",
        disp=True,
    )

    best_params = vector_to_params(result.x)
    dataset = generate_dataset(best_params, seed=seed)
    metrics = compute_metrics(dataset)
    return best_params, dataset, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic survey dataset")
    parser.add_argument("--optimize", action="store_true", help="Run differential evolution search before generating data")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.optimize:
        params, dataset, metrics = optimize_parameters(seed=args.seed)
    else:
        params = DEFAULT_PARAMS.copy()
        dataset = generate_dataset(params, seed=args.seed)
        metrics = compute_metrics(dataset)

    dataset.to_csv("data/project_survey.csv", index=False)
    print("\nBest parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")

    print("\nConstruct summaries:")
    summary_rows = []
    for construct in CONSTRUCT_ORDER:
        summary_rows.append({
            "Construct": construct,
            "Mean": metrics["means"][construct],
            "SD": metrics["sds"][construct],
            "Alpha": metrics["alpha"][construct],
        })
    print(pd.DataFrame(summary_rows))

    print("\nComposite correlation matrix:")
    composites = dataset[[f"{c}_mean" for c in CONSTRUCT_ORDER]]
    print(composites.corr())

    print("\nRegression coefficients:")
    print(metrics["regression"])

    print("\nMediation paths:")
    print(metrics["mediation"])

    print("\nModeration coefficients:")
    print(metrics["moderation"])
