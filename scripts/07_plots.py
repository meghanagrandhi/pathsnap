"""All comparison plots, generated only from the CSV/pickle outputs of the experiment scripts, plus genuine
epoch-wise training histories (LightGBM boosting rounds, MLP epochs) from a group-held-out validation split.

Usage: python scripts/07_plots.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.dataset import load_partition, xy_of  # noqa: E402

R, F = C.RESULT_DIR, C.FIG_DIR
os.makedirs(F, exist_ok=True)


def bar(df, col, title, fname, err=None, ylabel=None):
    d = df.dropna(subset=[col])
    plt.figure(figsize=(8, 4))
    plt.bar(d["model"], d[col], yerr=None if err is None else d[err], capsize=3, color="tab:blue")
    plt.xticks(rotation=35, ha="right"); plt.title(title); plt.ylabel(ylabel or col)
    plt.tight_layout(); plt.savefig(os.path.join(F, fname), dpi=130); plt.close()


def main():
    comp = pd.read_csv(os.path.join(R, "model_comparison.csv"))
    bar(comp, "cv_macro_f1_mean", "Group-CV Macro F1 (train trips, mean ± std over folds)", "cmp_cv_macro_f1.png", "cv_macro_f1_std")
    bar(comp, "test_top1", "Student-test observation Top-1 accuracy", "cmp_test_top1.png")
    bar(comp, "test_macro_f1", "Student-test candidate Macro F1", "cmp_test_macro_f1.png")
    bar(comp, "test_log_loss", "Student-test Log Loss (lower is better)", "cmp_test_logloss.png")
    bar(comp, "inference_ms_per_obs", "Inference time (ms per GPS observation)", "cmp_inference_time.png")
    # fold-level box plot
    folds = pd.read_csv(os.path.join(R, "cv_fold_scores.csv"))
    names = list(folds["model"].unique())
    plt.figure(figsize=(8, 4))
    plt.boxplot([folds.loc[folds.model == n, "macro_f1"] for n in names], tick_labels=names)
    plt.xticks(rotation=35, ha="right"); plt.title("Macro F1 across CV folds"); plt.tight_layout()
    plt.savefig(os.path.join(F, "cmp_cv_fold_boxplot.png"), dpi=130); plt.close()
    # confusion matrices (student test)
    cms = [(n, pd.read_csv(os.path.join(R, f"confusion_{n}.csv"), index_col=0)) for n in names]
    fig, axes = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.2))
    for ax, (n, cm) in zip(np.atleast_1d(axes), cms):
        ax.imshow(cm.values, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm.values):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8)
        ax.set_title(n, fontsize=8); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("Candidate-level confusion matrices, student-test partition"); fig.tight_layout()
    fig.savefig(os.path.join(F, "confusion_matrices.png"), dpi=130); plt.close(fig)

    tv = os.path.join(R, "tuned_vs_untuned.csv")
    if os.path.exists(tv):
        t = pd.read_csv(tv)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
        for a, c in zip(ax, ["cv_macro_f1_mean", "test_top1", "test_log_loss"]):
            a.bar(t["model"], t[c]); a.set_title(c); a.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(os.path.join(F, "tuned_vs_untuned.png"), dpi=130); plt.close(fig)
    bt = os.path.join(R, "bayes_trials.csv")
    if os.path.exists(bt):
        b = pd.read_csv(bt)
        plt.figure(figsize=(6, 4)); plt.plot(b["trial"], b["objective"], "o-", label="trial")
        plt.plot(b["trial"], b["objective"].cummax(), "r-", label="best so far")
        plt.xlabel("Optuna trial"); plt.ylabel("objective (CV)"); plt.legend(); plt.title("Bayesian optimisation history")
        plt.tight_layout(); plt.savefig(os.path.join(F, "bayes_history.png"), dpi=130); plt.close()
    hc = os.path.join(R, "hmm_comparison.csv")
    if os.path.exists(hc):
        h = pd.read_csv(hc)
        cols = [c for c in ["top1", "route_f1", "cont_strict", "cont_relaxed"] if c in h]
        h.set_index("system")[cols].plot.bar(figsize=(8, 4), rot=0); plt.title("Classifier-only vs classifier + HMM (student test)")
        plt.ylim(0, 1.02); plt.tight_layout(); plt.savefig(os.path.join(F, "hmm_comparison.png"), dpi=130); plt.close()
        hp = os.path.join(R, "hmm_per_trip.csv")
        if os.path.exists(hp):
            p = pd.read_csv(hp, dtype={"trip": str}).pivot(index="trip", columns="system", values="acc")
            p.plot.bar(figsize=(9, 4)); plt.ylabel("Top-1 accuracy"); plt.title("Per-trip accuracy, student test")
            plt.tight_layout(); plt.savefig(os.path.join(F, "hmm_per_trip.png"), dpi=130); plt.close()

    training_curves()


def training_curves():
    """Real epoch-wise histories on an 80/20 trip-level split of the training partition."""
    from lightgbm import LGBMClassifier, log_evaluation
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from pathsnap.models import LGBM_DEFAULT

    tr, _ = load_partition("train", R, training=True)
    X, y = xy_of(tr)
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=C.SEED).split(X, y, tr["trip"]))
    m = LGBMClassifier(**LGBM_DEFAULT, random_state=C.SEED, verbose=-1)
    m.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], eval_names=["train", "val"], eval_metric="binary_logloss")
    ev = m.evals_result_
    plt.figure(figsize=(6, 4)); plt.plot(ev["train"]["binary_logloss"], label="train"); plt.plot(ev["val"]["binary_logloss"], label="validation (held-out trips)")
    plt.xlabel("boosting round"); plt.ylabel("log loss"); plt.legend(); plt.title("LightGBM training history")
    plt.tight_layout(); plt.savefig(os.path.join(F, "curve_lightgbm.png"), dpi=130); plt.close()

    sc = StandardScaler().fit(X[a])
    Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
    net = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, random_state=C.SEED)
    tl, vl = [], []
    for ep in range(30):
        net.partial_fit(Xa, y[a], classes=[0, 1])
        tl.append(log_loss(y[a], net.predict_proba(Xa)[:, 1], labels=[0, 1]))
        vl.append(log_loss(y[b], net.predict_proba(Xb)[:, 1], labels=[0, 1]))
    plt.figure(figsize=(6, 4)); plt.plot(tl, label="train"); plt.plot(vl, label="validation (held-out trips)")
    plt.xlabel("epoch"); plt.ylabel("log loss"); plt.legend(); plt.title("MLP training history")
    plt.tight_layout(); plt.savefig(os.path.join(F, "curve_mlp.png"), dpi=130); plt.close()
    pd.DataFrame(dict(epoch=range(len(tl)), train=tl, val=vl)).to_csv(os.path.join(R, "mlp_history.csv"), index=False)
    pd.DataFrame(dict(round=range(len(ev["train"]["binary_logloss"])), train=ev["train"]["binary_logloss"],
                      val=ev["val"]["binary_logloss"])).to_csv(os.path.join(R, "lightgbm_history.csv"), index=False)


if __name__ == "__main__":
    main()
