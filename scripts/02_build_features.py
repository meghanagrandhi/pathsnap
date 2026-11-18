"""Build candidate/feature tables (+ derived reference labels) for the train and student_test partitions.

Usage: python scripts/02_build_features.py --data-root <dir containing train/ and student_test/> [--limit N]
Outputs: results/features_<partition>.pkl, results/trip_stats_<partition>.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.features import build_candidate_table  # noqa: E402
from pathsnap.io import list_trip_dirs, load_trip  # noqa: E402
from pathsnap.labels import align_obs_to_route  # noqa: E402
from pathsnap.network import Network  # noqa: E402


def process(folder, radius, max_k):
    trip = load_trip(folder, with_route=True)
    net = Network(trip.nodes, trip.arcs, C.SAMPLE_STEP_M)
    df = build_candidate_table(trip, net, radius=radius, max_k=max_k)
    xy = net.project(trip.track[:, :2])
    labels, ridx, dlab = align_obs_to_route(net, xy, trip.route)
    df["y"] = (df["arc"].to_numpy() == labels[df["obs"].to_numpy()]).astype(np.int8)
    df["label_arc"] = labels[df["obs"].to_numpy()]
    df.insert(0, "trip", trip.trip_id)
    hit = df.groupby("obs")["y"].max()
    hit = hit.reindex(np.arange(len(xy)), fill_value=0)
    df["obs_hit"] = hit.to_numpy()[df["obs"].to_numpy()].astype(np.int8)  # reference arc is among candidates
    stats = dict(trip=trip.trip_id, n_obs=len(xy), n_rows=len(df), n_nodes=len(trip.nodes), n_arcs=len(trip.arcs),
                 route_len=len(trip.route), miss_rate=float(1 - hit.mean()), mean_label_dist=float(dlab.mean()),
                 p90_label_dist=float(np.percentile(dlab, 90)), p99_label_dist=float(np.percentile(dlab, 99)),
                 max_label_dist=float(dlab.max()), max_arc_len=float(net.length.max()),
                 median_arc_len=float(np.median(net.length)), mean_cands=float(df.groupby("obs").size().mean()),
                 track_path_m=float(np.hypot(*np.diff(xy, axis=0).T).sum()), route_m=float(net.length[trip.route].sum()))
    stats["usable"] = int(stats["mean_label_dist"] <= C.QC_MAX_MEAN_LABEL_DIST and stats["miss_rate"] <= C.QC_MAX_MISS)
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--radius", type=float, default=C.CAND_RADIUS_M)
    ap.add_argument("--max-k", type=int, default=C.MAX_CANDIDATES)
    ap.add_argument("--limit", type=int, default=0, help="only first N trips (quick tests)")
    ap.add_argument("--out", default=C.RESULT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for part in args.partitions:
        dirs = list_trip_dirs(os.path.join(args.data_root, part))
        if args.limit:
            dirs = dirs[: args.limit]
        frames, stats = [], []
        t0 = time.time()
        for k, d in enumerate(dirs):
            df, st = process(d, args.radius, args.max_k)
            frames.append(df)
            stats.append(st)
            print(f"[{part}] {k + 1}/{len(dirs)} trip {st['trip']}: obs={st['n_obs']} rows={st['n_rows']} "
                  f"miss={st['miss_rate']:.3f} p90d={st['p90_label_dist']:.1f}m  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(frames, ignore_index=True).to_pickle(os.path.join(args.out, f"features_{part}.pkl"))
        pd.DataFrame(stats).to_csv(os.path.join(args.out, f"trip_stats_{part}.csv"), index=False)


if __name__ == "__main__":
    main()
