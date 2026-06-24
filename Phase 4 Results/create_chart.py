
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# Configuration 

TOP_N = 20  # number of teams to display

# Stage colors (dark = deeper run)
STAGE_COLORS = {
    "Win":   "#0f2b50",
    "Final": "#1b4d8a",
    "SF":    "#2e73b8",
    "QF":    "#5a9fd4",
    "R16":   "#92c3e8",
    "R32":   "#cce1f4",
}

STAGE_LABELS = {
    "Win":   "Win tournament",
    "Final": "Reach final",
    "SF":    "Reach semifinal",
    "QF":    "Reach quarterfinal",
    "R16":   "Reach round of 16",
    "R32":   "Reach round of 32 only",
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the probabilities CSV."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("P(Champion)", ascending=False).head(TOP_N)

    # Compute incremental probabilities for stacking
    df["Win"]   = df["P(Champion)"]
    df["Final"] = df["P(Final)"]   - df["P(Champion)"]
    df["SF"]    = df["P(SF)"]      - df["P(Final)"]
    df["QF"]    = df["P(QF)"]      - df["P(SF)"]
    df["R16"]   = df["P(R16)"]     - df["P(QF)"]
    df["R32"]   = df["P(R32)"]     - df["P(R16)"]

    return df.reset_index(drop=True)


def create_chart(df: pd.DataFrame, output_path: str):
    """Create the stacked horizontal bar chart."""
    teams = df["Team"].tolist()[::-1]  # reverse for top-down display
    stages = ["Win", "Final", "SF", "QF", "R16", "R32"]
    y_pos = np.arange(len(teams))

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # Draw stacked bars 
    left = np.zeros(len(teams))

    for stage in stages:
        values = df[stage].tolist()[::-1]  # reverse to match teams
        values_pct = [v * 100 for v in values]

        bars = ax.barh(
            y_pos, values_pct, left=left * 100,
            height=0.65,
            color=STAGE_COLORS[stage],
            label=STAGE_LABELS[stage],
            edgecolor="white",
            linewidth=0.3,
        )

        # Add percentage labels on the "Win" segment for top teams
        if stage == "Win":
            for i, (bar, val) in enumerate(zip(bars, values_pct)):
                if val >= 2.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8, fontweight="600",
                        color="white",
                    )

        left += np.array(values)

    # Cumulative % label at end of each bar
    for i, team in enumerate(teams):
        row = df[df["Team"] == team].iloc[0]
        total = row["P(R32)"] * 100
        ax.text(
            total + 0.8, i,
            f"{total:.0f}%",
            ha="left", va="center",
            fontsize=9, color="#666",
        )

    # Axes formatting 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams, fontsize=12, fontweight="500")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Probability of reaching each stage", fontsize=11, color="#555")
    ax.tick_params(axis="x", colors="#888", labelsize=10)
    ax.tick_params(axis="y", length=0)

    # Grid
    ax.xaxis.grid(True, linestyle="-", alpha=0.15, color="#000")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Remove spines
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#ddd")

    # Title and subtitle 
    fig.text(
        0.08, 0.96,
        "FIFA World Cup 2026: Winning Probabilities",
        fontsize=18, fontweight="700", color="#1a1a1a",
        transform=fig.transFigure,
    )
    fig.text(
        0.08, 0.935,
        "50,000 Monte Carlo simulations  •  Ensemble model",
        fontsize=10, color="#888",
        transform=fig.transFigure,
    )

    # Legend 
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        loc="lower right",
        fontsize=9,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#ddd",
        ncol=2,
        borderpad=1,
        columnspacing=1.5,
    )
    legend.get_frame().set_alpha(0.95)

    # Save 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f" Chart saved → {output_path}")


def main():
    # Determine input CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/results/wc2026_probs.csv"

    output_path = "C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/results/wc2026_probs_stacked_bar_chart.png"

    print("Loading data …")
    df = load_data(csv_path)

    print(f"Creating chart for top {TOP_N} teams …")
    create_chart(df, output_path)

    # Print summary
    top = df.iloc[0]
    print(f"\n  Favorite: {top['Team']} ({top['P(Champion)']*100:.1f}%)")
    print(f"  Top 3 combined: {df.head(3)['P(Champion)'].sum()*100:.1f}%")


if __name__ == "__main__":
    main()
