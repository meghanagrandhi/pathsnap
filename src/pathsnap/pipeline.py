"""End-to-end inference pipeline shared by the API and the offline evaluation scripts.

predict_trip()   uses ONLY .track/.nodes/.arcs (route is never loaded) -> predictions.
evaluate_trip()  is server-side only: loads the reference route, derives labels and scores predictions.
"""
import json
import os
import time

import numpy as np

from . import config as C
from .features import build_candidate_table, feature_matrix
from .hmm import fill_route, viterbi
from .io import load_trip
from .labels import align_obs_to_route
from .metrics import obs_accuracy, route_continuity, route_prf, top1_choice
from .network import Network


class ModelBundle:
    """Loads the final classifier from artifacts/ (native LightGBM text or a joblib pickle)."""

    def __init__(self, artifact_dir=C.ARTIFACT_DIR):
        with open(os.path.join(artifact_dir, "model_meta.json")) as f:
            self.meta = json.load(f)
        self.kind = self.meta["kind"]
        path = os.path.join(artifact_dir, self.meta["file"])
        if self.kind == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=path)
        else:
            import joblib
            self.model = joblib.load(path)
        hp = os.path.join(artifact_dir, "hmm_config.json")
        self.hmm_cfg = json.load(open(hp)) if os.path.exists(hp) else dict(C.HMM_DEFAULT)
        self.features = self.meta.get("features", C.FEATURES)
        if self.features != C.FEATURES:
            raise RuntimeError("Feature list in model_meta.json differs from config.FEATURES")

    def predict_proba(self, X):
        if self.kind == "lightgbm":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


def predict_trip(folder, bundle: ModelBundle, use_hmm=True):
    t0 = time.time()
    trip = load_trip(folder, with_route=False)
    net = Network(trip.nodes, trip.arcs, C.SAMPLE_STEP_M)
    t1 = time.time()
    df = build_candidate_table(trip, net)
    t2 = time.time()
    prob = bundle.predict_proba(feature_matrix(df))
    t3 = time.time()
    obs = df["obs"].to_numpy()
    arc = df["arc"].to_numpy()
    N = len(trip.track)
    clf_rows = top1_choice(obs, prob, N)
    xy = net.project(trip.track[:, :2])
    if use_hmm:
        rows = viterbi(net, xy, trip.track[:, 2], obs, arc, df["along"].to_numpy(), prob, bundle.hmm_cfg)
    else:
        rows = clf_rows
    t4 = time.time()
    pred = np.where(rows >= 0, arc[np.maximum(rows, 0)], -1)
    clf_pred = np.where(clf_rows >= 0, arc[np.maximum(clf_rows, 0)], -1)
    route = fill_route(net, pred)
    return dict(trip_id=trip.trip_id, trip=trip, net=net, df=df, prob=prob, xy=xy, pred=pred, clf_pred=clf_pred,
                route=route, timing=dict(load=t1 - t0, features=t2 - t1, classifier=t3 - t2, hmm=t4 - t3,
                                         total=t4 - t0))


def evaluate_trip(folder, res):
    """Server-side scoring of a prediction result against the reference route (never sent to a client)."""
    trip = load_trip(folder, with_route=True)
    if trip.route is None:
        return None
    net, xy = res["net"], res["xy"]
    labels, _, _ = align_obs_to_route(net, xy, trip.route)
    t = trip.track[:, 2]
    out = {}
    for name, pred in (("classifier", res["clf_pred"]), ("hmm", res["pred"])):
        acc = float(np.mean(pred == labels))
        s, r, n = route_continuity(net, pred, xy, t)
        out[name] = dict(top1_acc=acc, continuity_strict=s, continuity_relaxed=r, n_switches=n)
    for name, pred in (("classifier", res["clf_pred"]), ("hmm", res["pred"])):
        p, r, f = route_prf(fill_route(net, pred), trip.route)
        out[name].update(route_precision=p, route_recall=r, route_f1=f)
    out["n_obs"] = int(len(labels))
    return out
