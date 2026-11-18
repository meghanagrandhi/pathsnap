"""Dataset inventory + exploratory analysis (raw files only; no models).

Usage: python scripts/01_eda.py --data-root <dir with train/ and student_test/>
Outputs: results/eda_inventory_<partition>.csv, results/eda_summary.json, results/data_quality_notes.txt
         figures/eda_*.png, figures/folium_eda_<partition>_<trip>.html
Partition used for every output is stated in the file name.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.io import list_trip_dirs, load_trip  # noqa: E402
from pathsnap.network import Network  # noqa: E402


def inventory(part_dir, part):
    rows = []
    for d in list_trip_dirs(part_dir):
        tid = os.path.basename(d)
        files = {e: os.path.join(d, f) for f in os.listdir(d) for e in ("track", "nodes", "arcs", "route") if f.endswith("." + e)}
        tr = load_trip(d, with_route=True)
        net = Network(tr.nodes, tr.arcs, C.SAMPLE_STEP_M)
        t = tr.track[:, 2]
        dt = np.diff(t)
        xy = net.project(tr.track[:, :2])
        step = np.hypot(*np.diff(xy, axis=0).T)
        n_v = len(tr.nodes)
        adj = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n_v, n_v))
        ncomp, comp = connected_components(adj, directed=False)
        sizes = np.bincount(comp)
        isolated = int(((net.in_deg + net.out_deg) == 0).sum())
        bad_coord = int((~np.isfinite(tr.track[:, :2])).any(axis=1).sum()
                        + ((np.abs(tr.track[:, 0]) > 180) | (np.abs(tr.track[:, 1]) > 90)).sum())
        rows.append(dict(
            partition=part, trip=tid, files="|".join(sorted(files)), bytes_track=os.path.getsize(files["track"]),
            n_obs=len(tr.track), nan_values=int(np.isnan(tr.track).sum()), invalid_coords=bad_coord,
            duplicate_timestamps=int((dt == 0).sum()), non_monotonic_time=int((dt < 0).sum()),
            dt_median=float(np.median(dt)), dt_max=float(dt.max()), frac_dt_eq_1=float((dt == 1).mean()),
            n_gaps_gt_5s=int((dt > 5).sum()), max_step_m=float(step.max()), median_step_m=float(np.median(step)),
            duration_s=float(t[-1] - t[0]),
            lon_min=tr.track[:, 0].min(), lon_max=tr.track[:, 0].max(), lat_min=tr.track[:, 1].min(), lat_max=tr.track[:, 1].max(),
            extent_x_km=float(np.ptp(xy[:, 0]) / 1000), extent_y_km=float(np.ptp(xy[:, 1]) / 1000),
            n_nodes=n_v, n_arcs=len(tr.arcs), n_components=int(ncomp), largest_component_frac=float(sizes.max() / n_v),
            isolated_nodes=isolated, arcs_with_reverse_frac=float(net.has_rev.mean()), zero_length_arcs=int((net.length < 0.5).sum()),
            median_arc_len_m=float(np.median(net.length)), max_arc_len_m=float(net.length.max()),
            route_arcs=0 if tr.route is None else len(tr.route),
            route_connected_frac=float(np.mean(net.to[tr.route[:-1]] == net.frm[tr.route[1:]])) if tr.route is not None and len(tr.route) > 1 else np.nan,
            route_max_id_ok=bool(tr.route is not None and tr.route.max() < len(tr.arcs)),
        ))
        print(part, tid, "ok", flush=True)
    return pd.DataFrame(rows)


def make_plots(inv_all, data_root, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in inv_all.groupby("partition"):
        ax[0].hist(g["n_obs"], bins=25, alpha=0.6, label=part)
        ax[1].hist(np.log10(g["n_arcs"]), bins=25, alpha=0.6, label=part)
        ax[2].hist(g["dt_max"], bins=25, alpha=0.6, label=part)
    ax[0].set_xlabel("GPS observations per trip"); ax[1].set_xlabel("log10(road segments per trip)")
    ax[2].set_xlabel("max time gap in trip (s)"); ax[2].set_yscale("log")
    for a in ax:
        a.legend(); a.set_ylabel("trips")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_trip_size_distributions.png"), dpi=130); plt.close(fig)

    # missing-value / validity analysis
    cols = ["nan_values", "invalid_coords", "duplicate_timestamps", "non_monotonic_time", "n_gaps_gt_5s",
            "isolated_nodes", "zero_length_arcs"]
    miss = inv_all.groupby("partition")[cols].sum()
    miss.to_csv(os.path.join(C.RESULT_DIR, "eda_missing_value_summary.csv"))
    miss.T.plot.bar(figsize=(9, 4), logy=True); plt.title("Data-quality counts by partition (sum over trips)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_missing_values.png"), dpi=130); plt.close()

    # time-step distribution (training partition)
    steps = []
    tr_dirs = list_trip_dirs(os.path.join(data_root, "train"))
    for d in tr_dirs:
        steps.append(np.diff(load_trip(d, with_route=False).track[:, 2]))
    steps = np.concatenate(steps)
    vals, cnt = np.unique(np.clip(steps, 0, 10), return_counts=True)
    plt.figure(figsize=(6, 4)); plt.bar(vals, cnt / cnt.sum()); plt.yscale("log")
    plt.xlabel("time step between consecutive fixes (s, clipped at 10)"); plt.ylabel("fraction")
    plt.title(f"Sampling interval — train partition ({(steps == 1).mean():.1%} exactly 1 s)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_time_steps_train.png"), dpi=130); plt.close()

    # trajectory + road network for three train trips and two student-test trips
    picks = [("train", d) for d in tr_dirs[:3]] + [("student_test", d) for d in list_trip_dirs(os.path.join(data_root, "student_test"))[:2]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4))
    for ax, (part, d) in zip(axes, picks):
        tr = load_trip(d, with_route=True)
        net = Network(tr.nodes, tr.arcs, C.SAMPLE_STEP_M)
        xy = net.project(tr.track[:, :2])
        # draw only arcs near the track bbox for readability
        lo, hi = xy.min(0) - 150, xy.max(0) + 150
        m = (net.A[:, 0] > lo[0]) & (net.A[:, 0] < hi[0]) & (net.A[:, 1] > lo[1]) & (net.A[:, 1] < hi[1])
        idx = np.where(m)[0][::max(1, m.sum() // 15000)]
        segs = np.stack([net.A[idx], net.B[idx]], axis=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.4))
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.2, color="tab:blue", label="GPS")
        if tr.route is not None:
            rs = np.stack([net.A[tr.route], net.B[tr.route]], axis=1)
            ax.add_collection(LineCollection(rs, colors="tab:green", linewidths=1.2))
        ax.set_aspect("equal"); ax.set_title(f"{part} trip {os.path.basename(d)}", fontsize=9)
        ax.autoscale()
    fig.suptitle("GPS fixes (blue), road network (grey), reference route (green)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_trajectories_network.png"), dpi=130); plt.close(fig)


def folium_map(part, d, out_html):
    import folium
    tr = load_trip(d, with_route=True)
    net = Network(tr.nodes, tr.arcs, C.SAMPLE_STEP_M)
    ll = tr.track[:, [1, 0]]
    m = folium.Map(location=ll.mean(0).tolist(), zoom_start=14, tiles="OpenStreetMap")
    net_fg = folium.FeatureGroup("Road network (arcs near track)", show=False)
    xy = net.project(tr.track[:, :2])
    lo, hi = xy.min(0) - 100, xy.max(0) + 100
    mid = (net.A + net.B) / 2
    keep = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0][:6000]
    for a in keep:
        folium.PolyLine([tr.nodes[net.frm[a]][::-1].tolist(), tr.nodes[net.to[a]][::-1].tolist()], color="#888", weight=1).add_to(net_fg)
    net_fg.add_to(m)
    gps = folium.FeatureGroup("Raw GPS observations")
    for p in ll[:: max(1, len(ll) // 1500)]:
        folium.CircleMarker(p.tolist(), radius=2, color="blue", fill=True, fill_opacity=0.7).add_to(gps)
    gps.add_to(m)
    ref = folium.FeatureGroup("Reference route")
    for a in tr.route:
        folium.PolyLine([tr.nodes[net.frm[a]][::-1].tolist(), tr.nodes[net.to[a]][::-1].tolist()], color="green", weight=4).add_to(ref)
    ref.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    args = ap.parse_args()
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)
    inv = []
    for part in ("train", "student_test"):
        df = inventory(os.path.join(args.data_root, part), part)
        df.to_csv(os.path.join(C.RESULT_DIR, f"eda_inventory_{part}.csv"), index=False)
        inv.append(df)
    inv = pd.concat(inv, ignore_index=True)
    summary = {p: dict(trips=int(len(g)), total_obs=int(g.n_obs.sum()), median_obs=float(g.n_obs.median()),
                       min_obs=int(g.n_obs.min()), max_obs=int(g.n_obs.max()),
                       trips_with_gaps_gt_5s=int((g.n_gaps_gt_5s > 0).sum()),
                       trips_multi_component_network=int((g.n_components > 1).sum()),
                       trips_with_isolated_nodes=int((g.isolated_nodes > 0).sum()),
                       mean_arcs_with_reverse=float(g.arcs_with_reverse_frac.mean()))
               for p, g in inv.groupby("partition")}
    json.dump(summary, open(os.path.join(C.RESULT_DIR, "eda_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    make_plots(inv, args.data_root, C.FIG_DIR)
    for part in ("train", "student_test"):
        d = list_trip_dirs(os.path.join(args.data_root, part))[0]
        folium_map(part, d, os.path.join(C.FIG_DIR, f"folium_eda_{part}_{os.path.basename(d)}.html"))


if __name__ == "__main__":
    main()
