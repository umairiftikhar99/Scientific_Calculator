import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

N = 225

SUMMARY_ROWS = [
    {"Path": "AI → Team Collaboration", "β": 1.03, "SE": 0.08, "t": 12.88, "p-value": 0.000},
    {"Path": "Team Collaboration → Success", "β": 0.36, "SE": 0.07, "t": 5.14, "p-value": 0.000},
    {"Path": "AI → Success (direct effect)", "β": 0.09, "SE": 0.08, "t": 1.13, "p-value": 0.261},
    {"Path": "AI → Success (total effect)", "β": 0.70, "SE": 0.09, "t": 7.78, "p-value": 0.000},
]

INTERPRETATION = dedent(
    """
    The mediation results indicate several important findings:

    • AI adoption strongly predicts collaboration (β = 1.03, p < .001):
      The positive and significant relationship suggests that the use of AI tools
      enhances collaboration among team members. For instance, AI-based dashboards,
      communication platforms, and predictive analytics can improve information sharing,
      coordination, and trust within teams.

    • Collaboration predicts success (β = 0.36, p < .001):
      Collaboration has a direct and significant impact on project success. Teams that
      communicate openly, coordinate tasks, and resolve conflicts constructively are more
      likely to deliver projects on time, within budget, and to stakeholder satisfaction.

    • Direct effect of AI on success becomes insignificant (β = 0.09, p = .261):
      Once collaboration is included as a mediator, the direct effect of AI adoption on
      project success is no longer significant.

    • Total effect remains significant (β = 0.70, p < .001):
      The overall relationship between AI and project success remains significant, but the
      pathway is fully explained by collaboration.

    Taken together, these results demonstrate full mediation, meaning AI adoption
    enhances project success not directly, but through its ability to foster stronger
    collaboration within teams.
    """
).strip()


def build_summary_table() -> pd.DataFrame:
    """Return the mediation summary table as a DataFrame."""

    df = pd.DataFrame(SUMMARY_ROWS, columns=["Path", "β", "SE", "t", "p-value"])
    df.insert(0, "n", N)
    return df


def write_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_summary_table()
    csv_path = output_dir / "mediation_summary.csv"
    json_path = output_dir / "mediation_summary.json"
    interpretation_path = output_dir / "mediation_interpretation.txt"

    table.to_csv(csv_path, index=False)
    table.to_json(json_path, orient="records", indent=2)
    interpretation_path.write_text(INTERPRETATION + "\n")

    print(f"Saved mediation summary table to {csv_path}")
    print(f"Saved mediation summary JSON to {json_path}")
    print(f"Saved interpretation text to {interpretation_path}")


if __name__ == "__main__":
    default_output = Path("mediation_outputs")
    write_outputs(default_output)

    print("\nMediation Analysis Summary (n = 225):")
    print(build_summary_table().to_string(index=False))
    print("\nInterpretation:\n" + INTERPRETATION)
