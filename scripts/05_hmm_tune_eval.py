"""Tune HMM/Viterbi decoder parameters and compare classifier-only vs classifier+HMM.

Decoder parameters are tuned on OUT-OF-FOLD training probabilities (training trips only; never on
evaluator data), then reported on the student-test partition with the chosen parameters.

Usage: python scripts/05_hmm_tune_eval.py --model LightGBM [--tuned-probs] [--n-tune-trips 20]
Outputs: results/hmm_grid.csv, results/hmm_comparison.csv, results/hmm_per_trip.csv, artifacts/hmm_config.json
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.dataset import load_partition  # noqa: E402
from pathsnap.evaluate import TripCache, run, summarise  # noqa: E402

GRID = dict(beta=[3.0, 6.0, 12.0, 25.0], tau=[0.5, 1.0, 2.0], unreachable_penalty=[8.0, 20.0],
            dist_slack_m=[100.0, 200.0])


def attach(df, store, name):
    idx = store["index"]
    assert (idx["trip"].to_numpy() == df["trip"].to_numpy()).all() and (idx["obs"].to_numpy() == df["obs"].to_numpy()).all()
    return store["probs"][name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    ap.add_argument("--results", default=C.RESULT_DIR)
    ap.add_argument("--model", default="LightGBM")
    ap.add_argument("--oof", default="oof_probs.pkl", help="OOF probability file for decoder tuning")
    ap.add_argument("--student", default="student_probs.pkl", help="student-test probability file")
    ap.add_argument("--n-tune-trips", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-save-config", action="store_true")
    args = ap.parse_args()

    tr, _ = load_partition("train", args.results, training=True)
    te, _ = load_partition("student_test", args.results, training=False)
    p_tr = attach(tr, pickle.load(open(os.path.join(args.results, args.oof), "rb")), args.model)
    p_te = attach(te, pickle.load(open(os.path.join(args.results, args.student), "rb")), args.model)

    trips = sorted(tr["trip"].unique())
    step = max(1, len(trips) // args.n_tune_trips)
    tune_trips = set(trips[::step][: args.n_tune_trips])
    cache_tr, cache_te = TripCache(args.data_root, "train"), TripCache(args.data_root, "student_test")

    # --- decoder parameter search (train OOF) ---------------------------------------------------
    keys = list(GRID)
    rows = []
    base = run(tr, p_tr, cache_tr, C.HMM_DEFAULT, tune_trips, with_route_f1=False)
    clf_top1 = summarise(base["classifier"])["top1"]
    print(f"classifier-only top1 on tuning trips (OOF): {clf_top1:.4f}")
    t0 = time.time()
    for vals in itertools.product(*[GRID[k] for k in keys]):
        cfg = {**C.HMM_DEFAULT, **dict(zip(keys, vals))}
        r = summarise(run(tr, p_tr, cache_tr, cfg, tune_trips, with_route_f1=False)["hmm"])
        rows.append({**dict(zip(keys, vals)), "hmm_top1": r["top1"], "cont_strict": r["cont_strict"],
                     "cont_relaxed": r["cont_relaxed"]})
        print(f"{dict(zip(keys, vals))} -> top1={r['top1']:.4f} cont={r['cont_relaxed']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    grid = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    grid.to_csv(os.path.join(args.results, f"hmm_grid{args.tag}.csv"), index=False)
    best = {**C.HMM_DEFAULT, **{k: float(grid.iloc[0][k]) for k in keys}}
    print("best decoder params:", best)
    if not args.no_save_config:
        os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
        json.dump(best, open(os.path.join(C.ARTIFACT_DIR, "hmm_config.json"), "w"), indent=2)

    # --- final comparison on student-test (decoder fixed) ----------------------------------------
    res = run(te, p_te, cache_te, best)
    comp = []
    for name in ("classifier", "hmm"):
        comp.append(dict(system="Classifier only" if name == "classifier" else "Classifier + HMM/Viterbi",
                         **summarise(res[name])))
    pd.DataFrame(comp).to_csv(os.path.join(args.results, f"hmm_comparison{args.tag}.csv"), index=False)
    per = pd.DataFrame([{**{"system": k}, **r} for k, v in res.items() for r in v])
    per["acc"] = per["correct"] / per["n"]
    per.to_csv(os.path.join(args.results, f"hmm_per_trip{args.tag}.csv"), index=False)
    print(pd.DataFrame(comp).to_string())


if __name__ == "__main__":
    main()
