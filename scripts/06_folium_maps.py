"""Folium map of GPS observations, road network, reference route, classifier-only and HMM-predicted routes.

Usage: python scripts/06_folium_maps.py --partition student_test --trip 00000056 [--data-root ..]
Output: figures/folium_pred_<partition>_<trip>.html   (uses artifacts/ model; reference route is for display
        in the offline report only - the API never returns it)
"""
import argparse
import os
import sys

import folium
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.hmm import fill_route  # noqa: E402
from pathsnap.pipeline import ModelBundle, evaluate_trip, predict_trip  # noqa: E402


def poly(net, nodes, arcs):
    return [[[nodes[net.frm[a]][1], nodes[net.frm[a]][0]], [nodes[net.to[a]][1], nodes[net.to[a]][0]]] for a in arcs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(C.ROOT, ".."))
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trip", required=True)
    args = ap.parse_args()
    folder = os.path.join(args.data_root, args.partition, args.trip)
    bundle = ModelBundle()
    res = predict_trip(folder, bundle, use_hmm=True)
    met = evaluate_trip(folder, res)
    from pathsnap.io import load_trip
    tr = res["trip"]
    ref = load_trip(folder, with_route=True).route
    net = res["net"]
    ll = tr.track[:, [1, 0]]
    m = folium.Map(location=ll.mean(0).tolist(), zoom_start=14)
    xy = res["xy"]
    lo, hi = xy.min(0) - 60, xy.max(0) + 60
    mid = (net.A + net.B) / 2
    keep = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0][:8000]
    g = folium.FeatureGroup("Road network (grey)", show=False)
    for seg in poly(net, tr.nodes, keep):
        folium.PolyLine(seg, color="#999", weight=1).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Raw GPS observations (blue)")
    for p in ll[:: max(1, len(ll) // 2000)]:
        folium.CircleMarker(p.tolist(), radius=2, color="blue", fill=True, fill_opacity=0.7).add_to(g)
    g.add_to(m)
    layers = [("Reference route (green)", "green", ref, 7), ("Classifier-only route (orange)", "orange", fill_route(net, res["clf_pred"]), 5),
              ("Classifier + HMM route (red)", "red", res["route"], 3)]
    for name, col, arcs, w in layers:
        g = folium.FeatureGroup(name)
        for seg in poly(net, tr.nodes, arcs):
            folium.PolyLine(seg, color=col, weight=w, opacity=0.85).add_to(g)
        g.add_to(m)
    h, c = met["hmm"], met["classifier"]
    legend = (f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;border:1px solid #888;"
              f"font:12px sans-serif'><b>{args.partition} trip {args.trip}</b><br>"
              f"<span style='color:blue'>&#9679;</span> GPS &nbsp;<span style='color:green'>&#9644;</span> reference<br>"
              f"<span style='color:orange'>&#9644;</span> classifier-only (top1 {c['top1_acc']:.3f})<br>"
              f"<span style='color:red'>&#9644;</span> +HMM (top1 {h['top1_acc']:.3f})</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([ll.min(0).tolist(), ll.max(0).tolist()])
    out = os.path.join(C.FIG_DIR, f"folium_pred_{args.partition}_{args.trip}.html")
    os.makedirs(C.FIG_DIR, exist_ok=True)
    m.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
