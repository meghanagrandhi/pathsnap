"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model (LightGBM), group-aware CV on
the training trips; then refit, compare tuned vs untuned, and export the final model for the API.

Usage: python scripts/04_bayes_opt.py [--trials 30] [--folds 5] [--objective macro_f1|log_loss|top1]
Outputs: results/bayes_trials.csv, results/bayes_best_params.json, results/tuned_vs_untuned.csv,
         results/oof_tuned.pkl, results/student_tuned.pkl,
         artifacts/model.txt (native LightGBM text model), artifacts/model_meta.json
The student-test partition is NOT used to pick the search space, objective or hyper-parameters.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.dataset import load_partition, top1_accuracy, xy_of  # noqa: E402
from pathsnap.metrics import candidate_metrics  # noqa: E402
from pathsnap.models import LGBM_DEFAULT  # noqa: E402

SEARCH_SPACE = {
    "n_estimators": "int [100, 600]",
    "learning_rate": "log-uniform [0.01, 0.3]",
    "num_leaves": "int [15, 255] (log)",
    "min_child_samples": "int [10, 200] (log)",
    "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.4, 1.0]",
    "reg_lambda": "log-uniform [1e-3, 50]",
    "max_depth": "int [-1 (none), 4..16]",
}


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 16),
    )


def cv_eval(params, X, y, groups, folds, tr):
    oof = np.zeros(len(y))
    f1s, t1s = [], []
    for a, b in folds:
        m = LGBMClassifier(**params, random_state=C.SEED, n_jobs=-1, verbose=-1)
        m.fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
        f1s.append(candidate_metrics(y[b], oof[b])["macro_f1"])
        t1s.append(top1_accuracy(tr.iloc[b], oof[b]))
    return oof, float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(t1s)), candidate_metrics(y, oof)["log_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    ap.add_argument("--results", default=C.RESULT_DIR)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--objective", default="macro_f1", choices=["macro_f1", "log_loss", "top1"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--artifact-dir", default=C.ARTIFACT_DIR)
    args = ap.parse_args()

    tr, _ = load_partition("train", args.results, training=True)
    te, te_stats = load_partition("student_test", args.results, training=False)
    if args.quick:
        tr = tr[tr["trip"].isin(tr["trip"].unique()[:10])].reset_index(drop=True)
    X, y = xy_of(tr)
    Xte, yte = xy_of(te)
    groups = tr["trip"].to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(X, y, groups))  # identical folds to 03_train_compare.py
    n_obs_te = int(te_stats["n_obs"].sum())

    sign = -1.0 if args.objective == "log_loss" else 1.0
    records = []

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()
        _, f1m, f1s, t1, ll = cv_eval(params, X, y, groups, folds, tr)
        val = {"macro_f1": f1m, "log_loss": ll, "top1": t1}[args.objective]
        records.append(dict(trial=trial.number, **params, cv_macro_f1=f1m, cv_macro_f1_std=f1s, cv_top1=t1,
                            cv_log_loss=ll, objective=val, seconds=time.time() - t0))
        pd.DataFrame(records).to_csv(os.path.join(args.results, "bayes_trials.csv"), index=False)
        print(f"trial {trial.number}: {args.objective}={val:.5f} f1={f1m:.4f} top1={t1:.4f} ll={ll:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return sign * val

    study = optuna.create_study(direction="maximize" if sign > 0 else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=8))
    # seed the search with the untuned default so the comparison is fair and the baseline is a candidate
    study.enqueue_trial({**{k: v for k, v in LGBM_DEFAULT.items() if k not in ("subsample_freq", "reg_lambda")},
                         "reg_lambda": 1e-3, "max_depth": 12})
    t_opt = time.time()
    study.optimize(objective, n_trials=args.trials)
    opt_seconds = time.time() - t_opt
    best = {**study.best_params, "subsample_freq": 1}
    json.dump(dict(best_params=best, objective=args.objective, n_trials=args.trials, search_space=SEARCH_SPACE,
                   optimisation_seconds=opt_seconds, seed=C.SEED, folds=args.folds),
              open(os.path.join(args.results, "bayes_best_params.json"), "w"), indent=2)

    # --- tuned vs untuned (same folds, same partitions) -----------------------------------------
    comp, store_oof, store_te = [], {}, {}
    final_models = {}
    for label, params in (("LightGBM_untuned", dict(LGBM_DEFAULT)), ("LightGBM_tuned", best)):
        oof, f1m, f1s, t1, ll = cv_eval(params, X, y, groups, folds, tr)
        t0 = time.time()
        m = LGBMClassifier(**params, random_state=C.SEED, n_jobs=-1, verbose=-1).fit(X, y)
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        inf_s = time.time() - t0
        cm = candidate_metrics(yte, p)
        comp.append(dict(model=label, cv_macro_f1_mean=f1m, cv_macro_f1_std=f1s, cv_top1=t1, cv_log_loss=ll,
                         test_top1=top1_accuracy(te, p, n_obs_te), test_macro_f1=cm["macro_f1"],
                         test_log_loss=cm["log_loss"], train_time_s=fit_s, inference_time_s=inf_s,
                         inference_ms_per_obs=1000 * inf_s / n_obs_te))
        store_oof[label], store_te[label] = oof, p
        final_models[label] = m
        print(comp[-1])
    pd.DataFrame(comp).to_csv(os.path.join(args.results, "tuned_vs_untuned.csv"), index=False)
    pickle.dump(dict(index=tr[["trip", "obs", "arc"]].reset_index(drop=True), probs=store_oof),
                open(os.path.join(args.results, "oof_tuned.pkl"), "wb"))
    pickle.dump(dict(index=te[["trip", "obs", "arc"]].reset_index(drop=True), probs=store_te),
                open(os.path.join(args.results, "student_tuned.pkl"), "wb"))

    # --- export the tuned model for the API -------------------------------------------------------
    os.makedirs(args.artifact_dir, exist_ok=True)
    final_models["LightGBM_tuned"].booster_.save_model(os.path.join(args.artifact_dir, "model.txt"))
    import lightgbm
    json.dump(dict(kind="lightgbm", file="model.txt", features=C.FEATURES, params=best,
                   lightgbm_version=lightgbm.__version__, radius=C.CAND_RADIUS_M, max_k=C.MAX_CANDIDATES),
              open(os.path.join(args.artifact_dir, "model_meta.json"), "w"), indent=2)
    print("exported model to", args.artifact_dir)


if __name__ == "__main__":
    main()
