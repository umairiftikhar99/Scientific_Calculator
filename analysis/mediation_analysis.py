"""Emit the mediation analysis table matching the provided slide."""

from __future__ import annotations


def mediation_table() -> str:
    """Return the exact textual layout of the mediation slide."""

    lines = [
        "Mediation Analysis (Team Collaboration)",
        "Path\tβ\tt\tp\tResult",
        "AI → Collaboration\t1.03\t12.88\t.000\tSupported",
        "Collaboration → Success\t0.36\t5.14\t.000\tSupported",
        "AI → Success (Direct)\t0.09\t1.13\t.261\tNot Significant",
        "AI → Success (Total)\t0.70\t7.78\t.000\tSupported",
        "",
        "•Result: Full Mediation via Team Collaboration",
    ]
    return "\n".join(lines)


def main() -> None:
    print(mediation_table())


if __name__ == "__main__":
    main()
