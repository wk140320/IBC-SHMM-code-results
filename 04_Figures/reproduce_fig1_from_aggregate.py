#!/usr/bin/env python3
"""Recreate Figure 1 from the public KBS aggregate result table.

The right panel uses rounded means and sample SDs in the published aggregate CSV,
so the output is a rounded-data recreation, not a pixel-identical rebuild from
seed-level outputs. The left panel is explicitly illustrative asymptotic scaling.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = FIGURE_DIR.parent / "03_Results" / "paper_results_kbs_aggregate.csv"
DEFAULT_OUTPUT = FIGURE_DIR / "reproduced_from_aggregate"


def load_schur_summary(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = [
        r for r in rows
        if r["paper_result_id"].startswith("S04_Schur_rho")
        and r["metric"] == "FaultStep_TrueState_RMSE"
    ]
    if len(selected) != 4:
        raise ValueError(f"expected 4 S04 rho summary rows, found {len(selected)}")

    selected.sort(key=lambda r: float(r["scenario"].split("=", 1)[1]))
    rhos, schur_means, schur_sds, naive_means, naive_sds = [], [], [], [], []
    for r in selected:
        rho = float(r["scenario"].split("=", 1)[1])
        detail = r["comparison_or_p_value"]
        match = re.search(r"Naive=([0-9.]+)\s*±\s*([0-9.]+)", detail)
        if not match:
            raise ValueError(f"could not parse naive mean/SD for rho={rho}: {detail!r}")
        rhos.append(rho)
        schur_means.append(float(r["mean_value"]))
        schur_sds.append(float(r["sd_value"]))
        naive_means.append(float(match.group(1)))
        naive_sds.append(float(match.group(2)))
    return rhos, schur_means, schur_sds, naive_means, naive_sds


def build_figure(results_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    rhos, schur_means, schur_sds, naive_means, naive_sds = load_schur_summary(results_csv)
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), constrained_layout=True)

    ax = axes[0]
    delta = np.geomspace(1, 16, 200)
    ax.loglog(delta, np.ones_like(delta),
              label=r"Tier 2: healthy-channel weight $\Theta(1)$", lw=2)
    ax.loglog(delta, delta**-6,
              label=r"Contaminated-channel weight $\Theta(\delta^{-6})$ (Tier 1/2)", lw=2)
    ax.loglog(delta, delta**-4,
              label=r"Half-channel boundary: healthy weight $\Theta(\delta^{-4})$", lw=2)
    ax.set_title("(a) Theoretical weight rates (asymptotic, illustrative)")
    ax.set_xlabel(r"Contamination amplitude $\delta$ (normalized; illustrative scaling)")
    ax.set_ylabel("Normalized channel weight (illustrative)")
    ax.grid(True, which="both", ls=":", alpha=.45)
    ax.legend(loc="lower left", frameon=True)

    ax = axes[1]
    colors = ["#147d64", "#c43b3b"]
    series = [
        ("Schur-consistent variance-inflated", schur_means, schur_sds, "o", colors[0], colors[0]),
        ("Naive sandwich", naive_means, naive_sds, "s", "white", colors[1]),
    ]
    for label, means, sds, marker, face, color in series:
        ax.errorbar(rhos, means, yerr=sds, marker=marker, mfc=face,
                    mec=color, capsize=3, lw=2, label=label, color=color)
    ax.set_title("(b) Synthetic correlated-noise SLDS: fault-step state RMSE")
    ax.set_xlabel(r"Observation-noise correlation $\rho$")
    ax.set_ylabel("True latent-state RMSE (mean ± SD, 10 seeds)")
    ax.set_xticks(rhos)
    ax.grid(True, ls=":", alpha=.45)
    ax.legend(loc="upper left", frameon=True)
    ax.text(.02, .02,
            "Each point summarizes 10 generator seeds; lower is better.\n"
            "Plotted means/SDs are rounded in the public aggregate table.",
            transform=ax.transAxes, fontsize=7.2, va="bottom")
    fig.suptitle("Schur-consistent precision: theoretical boundary and synthetic evaluation", fontsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "fig1_kbs_mechanism_and_clean_weight_retention_reproduced.png"
    pdf = output_dir / "fig1_kbs_mechanism_and_clean_weight_retention_reproduced.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="path to the public aggregate result CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="output folder (defaults to a new subfolder under 04_Figures)")
    args = parser.parse_args()
    png, pdf = build_figure(args.results, args.output_dir)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
