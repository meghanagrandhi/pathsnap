"""Offline evaluation of classifier-only vs classifier+HMM decoding from stored candidate probabilities."""
import os

import numpy as np

from . import config as C
from .hmm import fill_route, viterbi
from .io import load_trip
from .metrics import route_continuity, route_prf, top1_choice
from .network import Network


class TripCache:
    def __init__(self, data_root, part):
        self.root, self.part, self._c = data_root, part, {}

    def get(self, trip_id):
        if trip_id not in self._c:
            tr = load_trip(os.path.join(self.root, self.part, trip_id), with_route=True)
            net = Network(tr.nodes, tr.arcs, C.SAMPLE_STEP_M)
            self._c[trip_id] = (tr, net, net.project(tr.track[:, :2]))
        return self._c[trip_id]


def decode_trip(df_trip, prob, tr, net, xy, hmm_cfg, with_route_f1=True):
    """df_trip: candidate rows of ONE trip (reset index, sorted by obs). Returns metric dict + predictions."""
    N = len(xy)
    obs = df_trip["obs"].to_numpy()
    arc = df_trip["arc"].to_numpy()
    lab = np.full(N, -1, dtype=np.int64)
    lab[obs] = df_trip["label_arc"].to_numpy()
    has_label = lab >= 0
    clf_rows = top1_choice(obs, prob, N)
    hmm_rows = viterbi(net, xy, tr.track[:, 2], obs, arc, df_trip["along"].to_numpy(), prob, hmm_cfg)
    out = {}
    for name, rows in (("classifier", clf_rows), ("hmm", hmm_rows)):
        pred = np.where(rows >= 0, arc[np.maximum(rows, 0)], -1)
        s, r, k = route_continuity(net, pred, xy, tr.track[:, 2])
        d = dict(correct=int(np.sum((pred == lab) & has_label)), n=int(has_label.sum()), cont_strict=s,
                 cont_relaxed=r, n_switch=k)
        if with_route_f1 and tr.route is not None:
            d["route_prec"], d["route_rec"], d["route_f1"] = route_prf(fill_route(net, pred), tr.route)
        out[name] = d
    return out


def run(df, prob, cache: TripCache, hmm_cfg, trips=None, with_route_f1=True):
    """Aggregate over trips. df must have columns trip, obs, arc, along, label_arc and be index-aligned to prob."""
    res = {"classifier": [], "hmm": []}
    for trip_id, g in df.groupby("trip", sort=False):
        if trips is not None and trip_id not in trips:
            continue
        tr, net, xy = cache.get(trip_id)
        idx = g.index.to_numpy()
        r = decode_trip(g.reset_index(drop=True), prob[idx], tr, net, xy, hmm_cfg, with_route_f1)
        for k in res:
            r[k]["trip"] = trip_id
            res[k].append(r[k])
    return res


def summarise(rows):
    n = sum(r["n"] for r in rows)
    sw = sum(r["n_switch"] for r in rows)
    out = dict(top1=sum(r["correct"] for r in rows) / max(n, 1),
               cont_strict=sum(r["cont_strict"] * r["n_switch"] for r in rows) / max(sw, 1),
               cont_relaxed=sum(r["cont_relaxed"] * r["n_switch"] for r in rows) / max(sw, 1))
    if "route_f1" in rows[0]:
        out["route_f1"] = float(np.mean([r["route_f1"] for r in rows]))
        out["route_prec"] = float(np.mean([r["route_prec"] for r in rows]))
        out["route_rec"] = float(np.mean([r["route_rec"] for r in rows]))
    return out
