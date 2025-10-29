import numpy as np

# Targets supplied in the request
means = np.array([3.79, 3.58, 3.19, 3.89])
sds = np.array([0.65, 0.72, 0.69, 0.71])
# Correlation matrix (AI, TC, PC, PS)
R = np.array([
    [1.00, 0.83, -0.12, 0.34],
    [0.83, 1.00, -0.15, 0.42],
    [-0.12, -0.15, 1.00, -0.25],
    [0.34, 0.42, -0.25, 1.00],
])

# Covariance matrix implied by correlations and standard deviations
cov = sds[:, None] * sds[None, :] * R

# Target regression coefficients (Intercept, AI, TC, PC)
regression_target = {
    "Intercept": 2.11,
    "AI": 0.21,
    "TC": 0.36,
    "PC": -0.18,
}

# Solve for the regression coefficients implied by the correlation matrix
Sxx = cov[:3, :3]
Sxy = cov[:3, 3]
coef_from_cov = np.linalg.solve(Sxx, Sxy)
intercept_from_cov = means[3] - coef_from_cov.dot(means[:3])

print("Coefficients implied by the correlation matrix:")
print(f"Intercept ≈ {intercept_from_cov:.3f}")
print(f"AI ≈ {coef_from_cov[0]:.3f}")
print(f"TC ≈ {coef_from_cov[1]:.3f}")
print(f"PC ≈ {coef_from_cov[2]:.3f}")

print("\nRequested coefficients:")
for key, value in regression_target.items():
    print(f"{key}: {value}")

# Show the difference (should be zero if the targets are mutually consistent)
diff = np.array([
    intercept_from_cov - regression_target["Intercept"],
    coef_from_cov[0] - regression_target["AI"],
    coef_from_cov[1] - regression_target["TC"],
    coef_from_cov[2] - regression_target["PC"],
])
print("\nDifferences (implied - requested):")
print(diff)

if np.any(np.abs(diff) > 1e-2):
    print("\nConclusion: the requested regression coefficients are incompatible with the provided\n"
          "correlation matrix and descriptive statistics. Any dataset that exactly matches\n"
          "the correlation table will necessarily yield the coefficients shown above when\n"
          "running the regression PS ~ AI + TC + PC.")
else:
    print("\nThe regression targets are numerically consistent with the correlation matrix.")
