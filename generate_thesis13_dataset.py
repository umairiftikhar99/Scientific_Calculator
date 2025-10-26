import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)

n = 225
means = np.array([3.83, 3.59, 3.18, 3.91])
sds = np.array([0.66, 0.75, 0.71, 0.73])
columns = ["AI_Mean", "TC_Mean", "PC_Mean", "PS_Mean"]
corr = np.array([
    [1.00,  0.812, -0.178,  0.449],
    [0.812, 1.00,  -0.206,  0.485],
    [-0.178, -0.206, 1.00, -0.282],
    [0.449,  0.485, -0.282, 1.00],
])

cov = np.outer(sds, sds) * corr

# Generate multivariate normal data with exact sample covariance
raw = np.random.standard_normal((n, len(columns)))
raw -= raw.mean(axis=0, keepdims=True)
S = raw.T @ raw / (n - 1)
eigvals, eigvecs = np.linalg.eigh(S)
white = raw @ (eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T)
L = np.linalg.cholesky(cov)
transformed = white @ L.T + means

df = pd.DataFrame(transformed, columns=columns)

# Embed moderation-friendly variation while preserving regression targets
gamma = 0.10
interaction = df["AI_Mean"] * df["PC_Mean"]
X_reg = sm.add_constant(df[["AI_Mean", "TC_Mean", "PC_Mean"]])
interaction_resid = sm.OLS(interaction, X_reg).fit().resid

# Add orthogonal residual component to PS_Mean
ps_adjusted = df["PS_Mean"] + gamma * interaction_resid

# Rescale to target mean and SD
ps_adjusted = (ps_adjusted - ps_adjusted.mean())
ps_adjusted = ps_adjusted * (sds[-1] / ps_adjusted.std(ddof=1)) + means[-1]

# Align regression coefficients with targets (Table 4.4)
reg_target = np.array([2.741, 0.171, 0.314, -0.194])
reg_model = sm.OLS(ps_adjusted, X_reg).fit()
delta = reg_target - reg_model.params.values
ps_adjusted = ps_adjusted + X_reg.values @ delta

df["PS_Mean"] = ps_adjusted

# Prepare outputs
summary = df.agg(['mean', 'std'])
correlation = df.corr()

regression = sm.OLS(df["PS_Mean"], X_reg).fit()
mediation_a = sm.OLS(df["TC_Mean"], sm.add_constant(df[["AI_Mean"]])).fit()
mediation_b = sm.OLS(df["PS_Mean"], sm.add_constant(df[["AI_Mean", "TC_Mean"]])).fit()
interaction_term = df["AI_Mean"] * df["PC_Mean"]
moderation = sm.OLS(df["PS_Mean"], sm.add_constant(pd.concat([
    df[["AI_Mean", "PC_Mean"]],
    interaction_term.rename("AI_PC")
], axis=1))).fit()

print("Descriptive Statistics (mean/std):")
print(summary)
print("\nCorrelation Matrix:")
print(correlation)
print("\nRegression Summary (PS_Mean ~ AI_Mean + TC_Mean + PC_Mean):")
print(regression.summary())
print("\nMediation Path Coefficients:")
print("AI -> TC (a path):", mediation_a.params)
print("AI + TC -> PS (b and c' paths):", mediation_b.params)
print("\nModeration Model Summary (PS_Mean ~ AI_Mean + PC_Mean + AI*PC):")
print(moderation.summary())

output = df.copy()
output.insert(0, "ParticipantID", np.arange(1, n + 1))
output.to_csv("Thesis13_synthetic.csv", index=False)
print("\nSynthetic dataset 'Thesis13_synthetic.csv' created successfully.")
