"""Train and compare >=5 models with group-aware (by trip) CV on the training partition, then evaluate each
on the student-test partition.

Usage: python scripts/03_train_compare.py [--data-root ..] [--quick] [--models LightGBM MLP ...]
Outputs (results/): model_comparison.csv, cv_fold_scores.csv, oof_probs.pkl, student_probs.pkl,
                    confusion_<model>.csv ; (figures/): comparison + confusion + loss-curve plots.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.dataset import load_partition, top1_accuracy, xy_of  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_continuity, top1_choice  # noqa: E402
from pathsnap.models import make_models  # noqa: E402
from pathsnap.network import Network  # noqa: E402


def predict_timed(model, X):
    t = time.time()
    p = model.predict_proba(X)[:, 1]
    return p, time.time() - t


def classifier_only_continuity(df, prob, data_root, part, cache):
    """Route continuity of the classifier-only (argmax) prediction, averaged over trips (weighted by switches)."""
    s_tot = r_tot = n_tot = 0
    for trip_id, g in df.groupby("trip", sort=False):
        if trip_id not in cache:
            tr = load_trip(os.path.join(data_root, part, trip_id), with_route=False)
            net = Network(tr.nodes, tr.arcs, C.SAMPLE_STEP_M)
            cache[trip_id] = (net, net.project(tr.track[:, :2]), tr.track[:, 2], len(tr.track))
        net, xy, t, n = cache[trip_id]
        p = prob[g.index.to_numpy()]
        rows = top1_choice(g["obs"].to_numpy(), p, n)
        arc = g["arc"].to_numpy()
        pred = np.where(rows >= 0, arc[np.maximum(rows, 0)], -1)
        s, r, k = route_continuity(net, pred, xy, t)
        s_tot += s * k
        r_tot += r * k
        n_tot += k
    return (s_tot / max(n_tot, 1), r_tot / max(n_tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    ap.add_argument("--results", default=C.RESULT_DIR)
    ap.add_argument("--quick", action="store_true", help="smoke test on a few trips")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tr, _ = load_partition("train", args.results, training=True)
    te, te_stats = load_partition("student_test", args.results, training=False)
    if args.quick:
        keep = tr["trip"].unique()[:10]
        tr = tr[tr["trip"].isin(keep)].reset_index(drop=True)
    X, y = xy_of(tr)
    Xte, yte = xy_of(te)
    groups = tr["trip"].to_numpy()
    n_obs_te = int(te_stats["n_obs"].sum())
    print(f"train rows={len(tr)} trips={tr['trip'].nunique()}  student-test rows={len(te)} trips={te['trip'].nunique()}")

    models = make_models()
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    gkf = GroupKFold(n_splits=args.folds)
    folds = list(gkf.split(X, y, groups))

    summary, fold_rows, oof_store, test_store, cache = [], [], {}, {}, {}
    out = args.results
    # resume/merge: keep results of models not re-run now (crash-safe, allows --models reruns)
    if os.path.exists(os.path.join(out, "model_comparison.csv")) and not args.quick:
        summary = [r for r in pd.read_csv(os.path.join(out, "model_comparison.csv")).to_dict("records")
                   if r["model"] not in models and r["model"] != "Baseline_NearestArc"]
        fold_rows = [r for r in pd.read_csv(os.path.join(out, "cv_fold_scores.csv")).to_dict("records") if r["model"] not in models]
        for fn, store in (("oof_probs.pkl", oof_store), ("student_probs.pkl", test_store)):
            if os.path.exists(os.path.join(out, fn)):
                old = pickle.load(open(os.path.join(out, fn), "rb"))["probs"]
                store.update({k: v for k, v in old.items() if k not in models})
    p_base = 1.0 / (1.0 + te["d"].to_numpy())

    def save_all():
        rows = summary + [dict(model="Baseline_NearestArc", family="Heuristic (no learning)",
                               config="argmax of 1/(1+d); no training", test_top1=top1_accuracy(te, p_base, n_obs_te))]
        pd.DataFrame(rows).to_csv(os.path.join(out, "model_comparison.csv"), index=False)
        pd.DataFrame(fold_rows).to_csv(os.path.join(out, "cv_fold_scores.csv"), index=False)
        with open(os.path.join(out, "oof_probs.pkl"), "wb") as f:
            pickle.dump(dict(index=tr[["trip", "obs", "arc"]].reset_index(drop=True), probs=oof_store), f)
        with open(os.path.join(out, "student_probs.pkl"), "wb") as f:
            pickle.dump(dict(index=te[["trip", "obs", "arc"]].reset_index(drop=True), probs=test_store), f)
    # baseline: nearest candidate (no learning)
    for name, (family, cfg, factory) in models.items():
        print(f"\n=== {name} ===", flush=True)
        oof = np.zeros(len(y))
        for k, (a, b) in enumerate(folds):
            t0 = time.time()
            m = factory()
            m.fit(X[a], y[a])
            oof[b] = m.predict_proba(X[b])[:, 1]
            cm = candidate_metrics(y[b], oof[b])
            t1 = top1_accuracy(tr.iloc[b], oof[b])
            fold_rows.append(dict(model=name, fold=k, macro_f1=cm["macro_f1"], log_loss=cm["log_loss"], top1=t1,
                                  fit_s=time.time() - t0))
            print(f"  fold {k}: macroF1={cm['macro_f1']:.4f} top1={t1:.4f} ll={cm['log_loss']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        folds_df = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        cv_all = candidate_metrics(y, oof)
        # final fit on the entire training partition
        t0 = time.time()
        m = factory()
        m.fit(X, y)
        fit_time = time.time() - t0
        p_te, inf_time = predict_timed(m, Xte)
        tm = candidate_metrics(yte, p_te)
        top1_te = top1_accuracy(te, p_te, n_obs_te)
        cont_s, cont_r = classifier_only_continuity(te, p_te, args.data_root, "student_test", cache)
        oof_store[name] = oof
        test_store[name] = p_te
        cmat = confusion_matrix(yte, (p_te >= 0.5).astype(int), labels=[0, 1])
        pd.DataFrame(cmat, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            os.path.join(args.results, f"confusion_{name}.csv"))
        summary.append(dict(
            model=name, family=family, config=cfg,
            cv_macro_f1_mean=folds_df["macro_f1"].mean(), cv_macro_f1_std=folds_df["macro_f1"].std(),
            cv_top1_mean=folds_df["top1"].mean(), cv_log_loss=cv_all["log_loss"],
            train_macro_f1=np.nan, test_macro_f1=tm["macro_f1"], test_precision=tm["precision"],
            test_recall=tm["recall"], test_top1=top1_te, test_log_loss=tm["log_loss"],
            test_route_continuity_strict=cont_s, test_route_continuity_relaxed=cont_r,
            train_time_s=fit_time, inference_time_s=inf_time, inference_ms_per_obs=1000 * inf_time / n_obs_te))
        # overfitting evidence: score on the data the final model was fitted on
        p_tr = m.predict_proba(X)[:, 1]
        summary[-1]["train_macro_f1"] = candidate_metrics(y, p_tr)["macro_f1"]
        summary[-1]["train_top1"] = top1_accuracy(tr, p_tr)
        print(f"  student-test: macroF1={tm['macro_f1']:.4f} top1={top1_te:.4f} ll={tm['log_loss']:.4f} "
              f"cont(strict/relaxed)={cont_s:.3f}/{cont_r:.3f} fit={fit_time:.0f}s", flush=True)
        save_all()

    save_all()
    json.dump(dict(seed=C.SEED, folds=args.folds, n_train_rows=len(tr), n_test_rows=len(te),
                   features=C.FEATURES, radius=C.CAND_RADIUS_M, max_k=C.MAX_CANDIDATES),
              open(os.path.join(out, "experiment_config.json"), "w"), indent=2)
    print("\nSaved results to", out)


if __name__ == "__main__":
    main()
