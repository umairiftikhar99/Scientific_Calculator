"""Reproduce the SPSS correlation table exactly as supplied by the user."""

from __future__ import annotations

def spss_correlation_table() -> str:
    """Return the literal SPSS-style table requested by the user."""

    lines = [
        "Correlations",
        "\t\tAI\tTC\tPC\tPS",
        "AI\tPearson Correlation\t1\t.830**\t-.120\t.340**",
        "\tSig. (2-tailed)\t\t.000\t.072\t.000",
        "\tN\t225\t225\t225\t225",
        "TC\tPearson Correlation\t.830**\t1\t-.150*\t.420**",
        "\tSig. (2-tailed)\t.000\t\t.024\t.000",
        "\tN\t225\t225\t225\t225",
        "PC\tPearson Correlation\t-.120\t-.150*\t1\t-.250**",
        "\tSig. (2-tailed)\t.072\t.024\t\t.000",
        "\tN\t225\t225\t225\t225",
        "PS\tPearson Correlation\t.340**\t.420**\t-.250**\t1",
        "\tSig. (2-tailed)\t.000\t.000\t.000\t",
        "\tN\t225\t225\t225\t225",
        "** Correlation is significant at the 0.01 level (2-tailed).",
        "* Correlation is significant at the 0.05 level (2-tailed).",
    ]
    return "\n".join(lines)


def main() -> None:
    print(spss_correlation_table())


if __name__ == "__main__":
    main()
