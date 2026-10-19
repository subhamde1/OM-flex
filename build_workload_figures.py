"""Build exact-data Figures 17 and 18 as vector PDFs and 600-dpi PNGs.

Run after workload_distortion.py. Uses only the package's synthetic output.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from workload_distortion import Model, F

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "replication/workload_distortion_results.json").read_text())
BLUE, ORANGE, NAVY, GREY = "#176b8d", "#ac4b16", "#112a4b", "#6b7380"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def finish(fig, number):
    fig.patch.set_facecolor("white")
    fig.add_artist(Rectangle((.008, .085), .984, .91,
                            transform=fig.transFigure, fill=False,
                            edgecolor="#52739c", lw=1))
    fig.text(.02, .025,
             "Source: Authors' fully synthetic calculations; Appendix I.\n"
             "Normalized contract quantities; no observed company data or electrical power.",
             fontsize=7.5, color="#33557d", va="bottom")
    fig.savefig(ROOT / f"Pic{number}.pdf")
    fig.savefig(ROOT / f"Pic{number}.png", dpi=600)
    plt.close(fig)

def amplification():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle("Workload bottlenecks can amplify capacity distortion",
                 y=.963, color=NAVY, weight="bold", fontsize=13)
    beta = np.linspace(0, 1, 1001)
    rr = .48 / (.48 + 2 * (beta-.6)**2)
    rq = beta/.6 * rr
    ax = axes[0]
    ax.axvspan(.6, 1, color="#eaf4ee", zorder=0)
    ax.axhline(1, color=GREY, lw=.9, ls=":")
    ax.plot(beta, rq, color=BLUE, lw=2, label="Capacity gap ratio")
    ax.plot(beta, rr, color=ORANGE, lw=2, label="Release gap ratio")
    for x in [1/3, .8]:
        y = .48/(.48+2*(x-.6)**2)
        ax.plot([x, x], [y, (x/.6)*y], color=GREY, lw=.8, ls=":")
        ax.scatter([x, x], [(x/.6)*y, y], color=[BLUE, ORANGE],
                   s=26, zorder=4)
    ax.text(.79, .29, "Capacity amplification\n$0.6<\\beta<1$",
            ha="center", fontsize=9, color=BLUE)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.27),
           xlabel=r"Binding workload slope $\beta$",
           ylabel="Ratio to workload-relaxed gap")
    ax.set_title("(a) Conditional on a common regime", pad=10, color=NAVY)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=.16)
    ax = axes[1]
    styles = [(GREY, "--", "Workload relaxed"),
              (BLUE, "-", "One event slot in three"),
              (ORANGE, "-", "Four event slots in five")]
    for case, (color, ls, label) in zip(D["cases"][:3], styles):
        ax.plot(case["theta"], np.asarray(case["distortions"])[:, 0],
                color=color, ls=ls, lw=2, label=label)
    ax.axvline(1.05, color=GREY, lw=.8, ls=":")
    for case, (color, _, _) in zip(D["cases"][:3], styles):
        ax.scatter([1.05], [case["selected_type"]["gaps"][0]["value"]],
                   s=25, color=color, zorder=4)
    ymax=max(np.asarray(case["distortions"])[:, 0].max()
             for case in D["cases"][:3])
    ax.set(xlim=(1, 1.1), ylim=(0, 1.10*ymax),
           xlabel=r"Private reservation-cost type $\theta$",
           ylabel=r"Capacity gap $q^{FB}-q^{SB}$")
    ax.set_title("(b) Complete mechanisms over all types", pad=10, color=NAVY)
    ax.legend(loc="upper left", fontsize=8.2, frameon=False)
    ax.grid(alpha=.16)
    fig.subplots_adjust(left=.075, right=.978, top=.82, bottom=.235, wspace=.30)
    finish(fig, 17)

def pooling():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle("A feasibility corner creates an endogenous service class",
                 y=.963, color=NAVY, weight="bold", fontsize=13)
    pool = D["pooling"]
    rstar, qstar = [pool[k]["value"] for k in ["r_star", "q_star"]]
    lo, hi = [pool[k]["value"] for k in ["w_plus", "w_minus"]]
    model = Model("One event slot in three", 3, 1, F("2.19"))
    r = np.linspace(.276, .3, 600)
    q = np.array([float(model.q(F(str(x)))) for x in r])
    ax = axes[0]
    ax.plot(r, .65+.6*r, color=GREY, ls=":", lw=1.8,
            label="Preferred unconstrained capacity")
    ax.plot(r, .73+r/3, color=BLUE, ls="--", lw=1.6,
            label="Workload bound")
    ax.plot(r, r/.35, color=ORANGE, ls="--", lw=1.6,
            label="Release-share bound")
    ax.plot(r, q, color=NAVY, lw=2.3, label="Conditional capacity")
    ax.scatter([rstar], [qstar], color=NAVY, s=32, zorder=5)
    ax.annotate(r"$(r_*,q_*)$", (rstar, qstar), xytext=(.277, .842),
                arrowprops={"arrowstyle": "->", "color": NAVY},
                color=NAVY, fontsize=10)
    ax.set(xlim=(.276, .30), ylim=(.809, .859),
           xlabel=r"Contractual release $r$",
           ylabel=r"Conditional capacity $q_{\rm H}(r)$")
    ax.set_title("(a) Local capacity requirements", pad=10, color=NAVY)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(alpha=.16)
    w = np.unique(np.r_[np.linspace(1, 1.1, 700), lo, hi])
    response = np.array([float(model.release(F(str(x)))) for x in w])
    ax = axes[1]
    ax.axvspan(lo, hi, color="#eaf4ee")
    ax.plot(w, response, color=BLUE, lw=2.2)
    ax.plot([lo, hi], [rstar, rstar], color=ORANGE, lw=3.4)
    ax.scatter([lo, hi], [rstar, rstar], color=ORANGE, s=25, zorder=5)
    ax.text((lo+hi)/2, .2808, "Pooling", ha="center", color=ORANGE,
            fontsize=10, weight="bold")
    ax.set(xlim=(1, 1.1), ylim=(.192, .301),
           xlabel=r"True or virtual cost argument $w$",
           ylabel=r"Optimal release $r_{\rm H}(w)$")
    ax.set_title("(b) Flat response over a cost interval", pad=10, color=NAVY)
    ax.grid(alpha=.16)
    fig.subplots_adjust(left=.082, right=.98, top=.82, bottom=.235, wspace=.30)
    finish(fig, 18)

if __name__ == "__main__":
    amplification()
    pooling()
    print("Built Figures 17 and 18: vector PDF and 600-dpi PNG.")
