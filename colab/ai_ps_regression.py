"""
Colab-ready script for performing regression analysis between AI and PS metrics
with a fixed significance threshold (alpha) of 0.007.

Usage steps in Google Colab:
1. Upload your dataset as a CSV file with columns named `AI` and `PS`.
2. Update the `DATA_SOURCE` variable to the path of the uploaded CSV.
3. Run the cells to obtain regression coefficients and p-values.
"""

import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------
# Provide the path to your CSV file uploaded in Colab. For example:
# DATA_SOURCE = "/content/drive/MyDrive/ai_ps_data.csv"
DATA_SOURCE = None  # Set to a string path to load your own data.

# In case you do not yet have a dataset, you can fall back to this example
# dictionary to see how the analysis works. Replace with your own values for
# actual research.
EXAMPLE_DATA = {
    "AI": [55, 60, 65, 70, 75, 80, 85, 90],
    "PS": [58, 62, 67, 71, 74, 78, 81, 85],
}

# Target significance level (alpha). The requirement is to use 0.007 instead of
# the more common 0.05 threshold.
ALPHA = 0.007

# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------
def load_data():
    """Load AI and PS values either from CSV or from the fallback example."""
    if DATA_SOURCE:
        df = pd.read_csv(DATA_SOURCE)
    else:
        df = pd.DataFrame(EXAMPLE_DATA)

    missing_columns = {"AI", "PS"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset must contain the columns: AI and PS. Missing "
            f"columns: {', '.join(sorted(missing_columns))}"
        )
    return df


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------
def run_regression(df: pd.DataFrame):
    """Run OLS regression and report statistics for the AI -> PS relation."""
    features = sm.add_constant(df[["AI"]], has_constant="add")
    target = df["PS"]

    model = sm.OLS(target, features).fit()
    return model


# ---------------------------------------------------------------------------
# Main execution flow when run as a script
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    print("Loaded dataset head:\n", df.head(), "\n")

    model = run_regression(df)

    ai_p_value = model.pvalues.get("AI", float("nan"))
    print(model.summary())
    print("\nAI coefficient p-value:", ai_p_value)

    if ai_p_value < ALPHA:
        print(
            f"Result: Significant at alpha={ALPHA}. The AI-PS relationship is "
            "statistically significant under the specified threshold."
        )
    else:
        print(
            f"Result: Not significant at alpha={ALPHA}. Consider collecting "
            "more data or reassessing the model specification."
        )


if __name__ == "__main__":
    main()
