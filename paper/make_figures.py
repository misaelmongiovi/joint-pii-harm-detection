"""Genera le figure del paper dai file del run (nessun numero scritto a mano).

    python paper/make_figures.py --run_dir /runs/20261016-1910_sensitive-xlmr-large --out paper/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette di riferimento (slot 1-3, validati anche all-pairs), testo e griglia recessivi.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})
COL = 3.45  # larghezza di una colonna IEEE (pollici)


def curves(rd, out):
    L = [json.loads(x) for x in open(os.path.join(rd, "metrics.jsonl"))]
    ev = {}
    for d in L:
        if "val/score" in d:
            ev[d["step"]] = d  # dedup dell'ultima valutazione ripetuta
    steps = sorted(ev)
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    series = [("val/pii/entity_f1_macro", "PII entity macro-F1", BLUE, "o", "-"),
              ("val/safety/unsafe_f1", "Unsafe F1", AQUA, "s", "--"),
              ("val/safety/cat_f1_macro", "Harm category macro-F1", ORANGE, "^", "-.")]
    for k, name, c, mk, ls in series:
        ys = [ev[s][k] for s in steps]
        ax.plot(steps, ys, color=c, lw=1.5, ls=ls, marker=mk, ms=4, label=name)
    spe = 2256
    for e in (1, 2):
        ax.axvline(e * spe, color=INK2, lw=0.6, ls=":")
        ax.text(e * spe + 60, 0.97, f"epoch {e + 1}", color=INK2, fontsize=6.5, va="top")
    ax.set_xlabel("Training step")
    ax.set_ylabel("F1 (validation)")
    ax.set_ylim(0.15, 1.0)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "val_curves.pdf"))


def categories(rd, out):
    r = json.load(open(os.path.join(rd, "test_results.json")))
    tv, t5 = r["thresholds_val"]["safety/per_category"], r["threshold_0.5"]["safety/per_category"]
    names = sorted(tv, key=lambda k: tv[k]["f1"])
    fig, ax = plt.subplots(figsize=(COL, 4.0))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], [tv[n]["f1"] for n in names], height=0.38, color=BLUE, label="Validation-tuned thresholds")
    ax.barh([i - 0.2 for i in y], [t5[n]["f1"] for n in names], height=0.38, color=ORANGE, label="Fixed threshold 0.5")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n} (n={tv[n]['support']})" for n in names], fontsize=6.5)
    ax.set_xlabel("F1 (test)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "safety_categories.pdf"))


def pii_types(rd, out):
    a = json.load(open(os.path.join(rd, "analysis", "analysis_test.json")))
    en, it = a["pii_en"]["per_type_f1"], a["pii_it"]["per_type_f1"]
    names = sorted(en, key=lambda k: (en[k] + it[k]) / 2)
    fig, ax = plt.subplots(figsize=(COL, 3.4))
    y = list(range(len(names)))
    for i, n in enumerate(names):
        ax.plot([en[n], it[n]], [i, i], color=GRID, lw=1.5, zorder=1)
    ax.scatter([en[n] for n in names], y, s=22, color=BLUE, marker="o", label="English", zorder=2, edgecolors="white", linewidths=0.6)
    ax.scatter([it[n] for n in names], y, s=22, color=AQUA, marker="s", label="Italian", zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("Entity-level F1 (test)")
    ax.set_xlim(0.65, 1.01)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "pii_types.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curves(a.run_dir, a.out); categories(a.run_dir, a.out); pii_types(a.run_dir, a.out)
    print("ok")
