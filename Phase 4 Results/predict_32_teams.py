
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# Group definitions 

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def load_data(csv_path):
    """Load simulation results CSV."""
    df = pd.read_csv(csv_path)
    if "P(R32)" not in df.columns:
        raise ValueError("CSV must contain P(R32) column")
    return df


def get_status(p):
    """Return status label based on advancement probability."""
    if p >= 0.85:   return "Lock"
    elif p >= 0.70: return "Likely"
    elif p >= 0.50: return "Toss-up"
    elif p >= 0.30: return "Tough"
    else:           return "Unlikely"


def print_groups(df):
    """Print group-by-group advancement probabilities."""
    print("=" * 72)
    print("  FIFA World Cup 2026 — Group Stage Advancement Probabilities")
    print("  Based on 50,000 Monte Carlo simulations | Full ensemble model")
    print("=" * 72)

    for g in sorted(GROUPS.keys()):
        teams = GROUPS[g]
        g_data = []
        for t in teams:
            row = df[df["Team"] == t]
            if len(row):
                p = row.iloc[0]["P(R32)"]
                g_data.append((t, p))

        g_data.sort(key=lambda x: -x[1])

        print(f"\n  Group {g}")
        print(f"  {'─' * 62}")

        for t, p in g_data:
            bar_len = int(p * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            status = get_status(p)
            exit_p = 1 - p
            print(f"  {t:<28s} {bar} {p:>5.1%}  ({status})")

        # Group competitiveness
        probs = [p for _, p in g_data]
        spread = max(probs) - min(probs)
        if spread < 0.20:
            print(f"  {'':>28s} ↑ Extremely competitive (spread {spread:.0%})")
        elif min(probs) > 0.50:
            print(f"  {'':>28s} ↑ All four teams have a real chance")


def print_32_qualifiers(df):
    """Print the 32 most likely qualifying teams."""
    all_teams = []
    for g, teams in GROUPS.items():
        for t in teams:
            row = df[df["Team"] == t]
            if len(row):
                all_teams.append({
                    "Team": t,
                    "Group": g,
                    "P(Advance)": row.iloc[0]["P(R32)"],
                    "P(Win WC)": row.iloc[0].get("P(Champion)", 0),
                })

    teams_df = pd.DataFrame(all_teams).sort_values("P(Advance)", ascending=False)
    teams_df = teams_df.reset_index(drop=True)

    print(f"\n\n{'=' * 72}")
    print(f"  PREDICTED ROUND OF 32 QUALIFIERS")
    print(f"{'=' * 72}")
    print(f"\n  {'#':<4} {'Team':<28} {'Grp':>3} {'P(Advance)':>10} {'P(Win WC)':>10}  Status")
    print(f"  {'─' * 68}")

    for i, (_, r) in enumerate(teams_df.iterrows()):
        status = get_status(r["P(Advance)"])

        if i == 24:
            print(f"  {'─' * 68}")
            print(f"  {'':>4} {'--- Auto-qualify line (top 2 per group) ---':<50}")
            print(f"  {'─' * 68}")
        if i == 32:
            print(f"  {'─' * 68}")
            print(f"  {'':>4} {'--- Best 3rd place cutoff ---':<50}")
            print(f"  {'─' * 68}")

        marker = "✓" if i < 32 else " "
        print(f"  {marker} {i+1:<3} {r['Team']:<28} {r['Group']:>3} "
              f"{r['P(Advance)']:>9.1%} {r['P(Win WC)']:>9.1%}  {status}")

        if i >= 39:
            remaining = len(teams_df) - 40
            print(f"\n  ... and {remaining} more teams below 40% advancement probability")
            break


def print_group_of_death(df):
    """Identify the most competitive groups."""
    print(f"\n\n{'=' * 72}")
    print(f"  GROUP OF DEATH ANALYSIS")
    print(f"{'=' * 72}")

    group_stats = []
    for g in sorted(GROUPS.keys()):
        probs = []
        for t in GROUPS[g]:
            row = df[df["Team"] == t]
            if len(row):
                probs.append(row.iloc[0]["P(R32)"])

        spread = max(probs) - min(probs)
        min_p = min(probs)
        avg_p = np.mean(probs)
        # Competitiveness score: low spread + high minimum = most competitive
        comp_score = (1 - spread) * min_p
        group_stats.append((g, spread, min_p, avg_p, comp_score))

    group_stats.sort(key=lambda x: -x[4])

    print(f"\n  {'Grp':>3} {'Spread':>8} {'Weakest':>9} {'Avg':>7} {'Verdict':<25}")
    print(f"  {'─' * 56}")

    for g, spread, min_p, avg_p, cs in group_stats:
        if spread < 0.20:
            verdict = "GROUP OF DEATH"
        elif min_p > 0.40 and spread < 0.40:
            verdict = "Very competitive"
        elif spread < 0.45:
            verdict = "Balanced"
        else:
            verdict = "Clear favorites"

        print(f"  {g:>3} {spread:>7.0%} {min_p:>8.0%} {avg_p:>6.0%}  {verdict}")


def print_summary(df):
    """Print key summary stats."""
    print(f"\n\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")

    # Safest teams
    all_p = []
    for g, teams in GROUPS.items():
        for t in teams:
            row = df[df["Team"] == t]
            if len(row):
                all_p.append((t, g, row.iloc[0]["P(R32)"]))

    all_p.sort(key=lambda x: -x[2])

    print(f"\n  Safest bets (>90% advance):")
    for t, g, p in all_p:
        if p >= 0.90:
            print(f"    {t} (Group {g}): {p:.1%}")

    print(f"\n  Biggest upset risks (strong teams <80%):")
    strong_teams = ["Germany", "France", "Brazil", "Netherlands",
                    "Belgium", "Uruguay", "Croatia", "Japan"]
    for t, g, p in all_p:
        if t in strong_teams and p < 0.80:
            print(f"    {t} (Group {g}): {p:.1%} — could be eliminated")

    print(f"\n  Cinderella candidates (underdogs >50%):")
    underdogs = ["Haiti", "Curaçao", "New Zealand", "Cape Verde",
                 "Panama", "Jordan", "Iraq", "Qatar", "Ghana",
                 "DR Congo", "Uzbekistan", "Saudi Arabia", "Paraguay"]
    for t, g, p in all_p:
        if t in underdogs and p >= 0.40:
            print(f"    {t} (Group {g}): {p:.1%}")


def generate_pdf(df, pdf_path):
    """Generate a polished PDF report with inline bar charts."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Flowable
    )

    # Custom inline bar chart flowable 
    class InlineBar(Flowable):
        """A small horizontal bar with percentage text overlaid."""
        def __init__(self, value, max_width=90, height=14,
                     bar_color=None, bg_color=None):
            Flowable.__init__(self)
            self.value = value  # 0.0 to 1.0
            self.max_width = max_width
            self.height = height
            self.width = max_width
            if bar_color is None:
                if value >= 0.85:
                    self.bar_color = colors.HexColor("#1b7a3d")
                elif value >= 0.70:
                    self.bar_color = colors.HexColor("#2e73b8")
                elif value >= 0.50:
                    self.bar_color = colors.HexColor("#e6a817")
                elif value >= 0.30:
                    self.bar_color = colors.HexColor("#d46b08")
                else:
                    self.bar_color = colors.HexColor("#c0392b")
            else:
                self.bar_color = bar_color
            self.bg_color = bg_color or colors.HexColor("#e8edf2")

        def draw(self):
            # Background track
            self.canv.setFillColor(self.bg_color)
            self.canv.roundRect(0, 0, self.max_width, self.height,
                                radius=3, fill=1, stroke=0)
            # Filled bar
            bar_w = max(self.value * self.max_width, 2)
            self.canv.setFillColor(self.bar_color)
            self.canv.roundRect(0, 0, bar_w, self.height,
                                radius=3, fill=1, stroke=0)
            # Text label
            text = f"{self.value:.1%}"
            if self.value >= 0.35:
                self.canv.setFillColor(colors.white)
                text_x = bar_w / 2
            else:
                self.canv.setFillColor(colors.HexColor("#333333"))
                text_x = bar_w + 4
            self.canv.setFont("Helvetica-Bold", 8)
            self.canv.drawString(text_x - 12, 3.5, text)

    # Document setup 
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#0f2b50"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.grey, spaceAfter=16, alignment=1,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#0f2b50"),
        spaceBefore=16, spaceAfter=8,
    )

    DARK = colors.HexColor("#0f2b50")
    MED = colors.HexColor("#2e73b8")

    story = []

    # Title 
    story.append(Paragraph("FIFA World Cup 2026", title_style))
    story.append(Paragraph("Group Stage Advancement Predictions", ParagraphStyle(
        "Sub", parent=styles["Heading3"], textColor=MED, alignment=1, spaceAfter=4,
    )))
    story.append(Paragraph(
        "Based on 50,000 Monte Carlo simulations  |  Full ensemble model (DC + XGBoost + LightGBM + MLP)",
        subtitle_style
    ))
    story.append(Spacer(1, 6))

    # Group-by-group tables with bar charts 
    for g in sorted(GROUPS.keys()):
        teams = GROUPS[g]
        g_data = []
        for t in teams:
            row = df[df["Team"] == t]
            if len(row):
                p = row.iloc[0]["P(R32)"]
                wc = row.iloc[0].get("P(Champion)", 0)
                g_data.append((t, p, wc))
        g_data.sort(key=lambda x: -x[1])

        story.append(Paragraph(f"Group {g}", h2_style))

        header = ["Team", "P(Advance)", "P(Win WC)", "Status"]
        table_data = [header]
        for t, p, wc in g_data:
            status = get_status(p)
            bar = InlineBar(p, max_width=100, height=14)
            table_data.append([t, bar, f"{wc:.1%}", status])

        tbl = Table(table_data, colWidths=[120, 110, 65, 65])

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 1), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (1, 1), (1, -1), 4),
        ]

        for i, (t, p, wc) in enumerate(g_data):
            row_idx = i + 1
            if p >= 0.85:
                style_cmds.append(("BACKGROUND", (0, row_idx), (0, row_idx), colors.HexColor("#e8f5e9")))
                style_cmds.append(("BACKGROUND", (2, row_idx), (-1, row_idx), colors.HexColor("#e8f5e9")))
            elif p >= 0.70:
                style_cmds.append(("BACKGROUND", (0, row_idx), (0, row_idx), colors.HexColor("#f1f8e9")))
                style_cmds.append(("BACKGROUND", (2, row_idx), (-1, row_idx), colors.HexColor("#f1f8e9")))
            elif p < 0.40:
                style_cmds.append(("BACKGROUND", (0, row_idx), (0, row_idx), colors.HexColor("#fce4ec")))
                style_cmds.append(("BACKGROUND", (2, row_idx), (-1, row_idx), colors.HexColor("#fce4ec")))

        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Round of 32 qualifiers with bar charts 
    story.append(Paragraph("Predicted Round of 32 Qualifiers", title_style))
    story.append(Spacer(1, 8))

    all_teams = []
    for g_, teams_ in sorted(GROUPS.items()):
        for t in teams_:
            row = df[df["Team"] == t]
            if len(row):
                r = row.iloc[0]
                all_teams.append((t, g_, r["P(R32)"], r.get("P(Champion)", 0)))
    all_teams.sort(key=lambda x: -x[2])

    header = ["#", "Team", "Grp", "P(Advance)", "P(Win WC)", "Status"]
    table_data = [header]
    for i, (t, g_, p, wc) in enumerate(all_teams[:36]):
        bar = InlineBar(p, max_width=80, height=12)
        table_data.append([str(i + 1), t, g_, bar, f"{wc:.1%}", get_status(p)])
        if i == 23:
            table_data.append(["", "--- Top 2 per group (auto-qualify) ---", "", "", "", ""])
        if i == 31:
            table_data.append(["", "--- Best 3rd place cutoff ---", "", "", "", ""])

    tbl = Table(table_data, colWidths=[22, 125, 25, 90, 55, 55])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ALIGN", (4, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (3, 1), (3, -1), 4),
    ]

    for i in range(len(table_data)):
        if i == 0: continue
        cell = table_data[i][1]
        if "---" in str(cell):
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff3cd")))
            style_cmds.append(("FONTNAME", (0, i), (-1, i), "Helvetica-BoldOblique"))
            style_cmds.append(("SPAN", (1, i), (4, i)))
            continue
        if i <= 24:
            style_cmds.append(("BACKGROUND", (0, i), (2, i), colors.HexColor("#e8f5e9")))
            style_cmds.append(("BACKGROUND", (4, i), (-1, i), colors.HexColor("#e8f5e9")))
        elif i <= 34:
            style_cmds.append(("BACKGROUND", (0, i), (2, i), colors.HexColor("#fff8e1")))
            style_cmds.append(("BACKGROUND", (4, i), (-1, i), colors.HexColor("#fff8e1")))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)

    doc.build(story)


def main():
    # Determine input CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try common output paths
        candidates = [
            "try_probs.csv",
            #"wc2026_win_probabilities.csv",
        ]
        csv_path = None
        for c in candidates:
            for d in [Path("."), Path("C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/simulation")]:
                if (d / c).exists():
                    csv_path = str(d / c)
                    break
            if csv_path:
                break

        if not csv_path:
            print("Usage: python predict_32_teams.py <probabilities.csv>")
            sys.exit(1)

    print(f"  Reading: {csv_path}\n")
    df = load_data(csv_path)

    # Print to console AND save to PDF 
    out_dir = Path(csv_path).parent
    pdf_path = out_dir / "wc2026_group_stage_predictions.pdf"

    print_groups(df)
    print_32_qualifiers(df)
    print_group_of_death(df)
    print_summary(df)

    # Generate PDF report
    print(f"\n  Generating PDF report …")
    generate_pdf(df, str(pdf_path))
    print(f"   PDF report saved → {pdf_path}")

    # Save detailed CSV 
    csv_out_path = out_dir / "wc2026_group_advancement_detailed.csv"
    rows = []
    for g in sorted(GROUPS.keys()):
        for t in GROUPS[g]:
            row = df[df["Team"] == t]
            if len(row):
                r = row.iloc[0]
                p = r["P(R32)"]
                rows.append({
                    "Group": g,
                    "Team": t,
                    "P(Advance)": round(p, 4),
                    "P(Exit Group)": round(1 - p, 4),
                    "P(Win WC)": round(r.get("P(Champion)", 0), 4),
                    "P(Reach Final)": round(r.get("P(Final)", 0), 4),
                    "P(Reach SF)": round(r.get("P(SF)", 0), 4),
                    "P(Reach QF)": round(r.get("P(QF)", 0), 4),
                    "Status": get_status(p),
                    "Qualifies": "Yes" if p >= 0.50 else "No",
                })

    out_df = pd.DataFrame(rows).sort_values(
        ["Group", "P(Advance)"], ascending=[True, False]
    )
    out_df.to_csv(csv_out_path, index=False)
    print(f"   Detailed CSV saved → {csv_out_path}")
    print(f"    ({len(out_df)} teams × {len(out_df.columns)} columns)")


if __name__ == "__main__":
    main()
