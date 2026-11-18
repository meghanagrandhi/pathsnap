"""PathSnap FastAPI backend (runs on Laptop 1).

Run:  PATHSNAP_TRIPS_DIR=<folder with the reserved trip folders> uvicorn api.main:app --host 0.0.0.0 --port 8000

Pipeline per request: load raw files (.track/.nodes/.arcs only) -> candidate generation -> features ->
classifier -> HMM/Viterbi -> route.  Only AFTER prediction does the server load the .route file (server-side
only) to score it.  Reference labels / route geometry are never returned to the client.
"""
import logging
import os
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pathsnap import config as C  # noqa: E402
from pathsnap.io import list_trip_dirs  # noqa: E402
from pathsnap.pipeline import ModelBundle, evaluate_trip, predict_trip  # noqa: E402

log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)

TRIPS_DIR = os.path.abspath(os.environ.get("PATHSNAP_TRIPS_DIR", os.path.join(ROOT, "..", "student_test")))
ARTIFACT_DIR = os.path.abspath(os.environ.get("PATHSNAP_ARTIFACTS", C.ARTIFACT_DIR))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle = None
_trips = {}
_lock = threading.Lock()


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def startup():
    global _bundle
    _bundle = ModelBundle(ARTIFACT_DIR)
    for d in list_trip_dirs(TRIPS_DIR):
        _trips[os.path.basename(d)] = d
    log.info("model=%s hmm_cfg=%s", _bundle.meta.get("kind"), _bundle.hmm_cfg)
    log.info("trips dir=%s  (%d trips: %s)", TRIPS_DIR, len(_trips), ", ".join(sorted(_trips)))


@app.get("/health")
def health():
    return {"status": "ok", "model": _bundle.meta.get("kind"), "n_trips": len(_trips)}


@app.get("/trips")
def trips():
    """Trip ids only (no reference information)."""
    out = []
    for tid, d in sorted(_trips.items()):
        out.append({"trip_id": tid})
    return {"trips": out}


def _polyline(net, nodes_ll, arcs):
    """[lat, lon] polyline through the given arcs (consecutive arcs share nodes where connected)."""
    pts = []
    for a in arcs:
        f, t = int(net.frm[a]), int(net.to[a])
        if not pts:
            pts.append(nodes_ll[f])
        elif not np.allclose(pts[-1], nodes_ll[f]):
            pts.append(nodes_ll[f])
        pts.append(nodes_ll[t])
    return [[float(p[1]), float(p[0])] for p in pts]


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id
    if tid not in _trips:
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid}")
    folder = _trips[tid]
    t0 = time.time()
    log.info("POST /predict trip_id=%s  -> running pipeline", tid)
    with _lock:  # one heavy request at a time keeps timings honest on a laptop
        res = predict_trip(folder, _bundle, use_hmm=True)
        metrics = evaluate_trip(folder, res)  # server-side reference scoring
    trip, net = res["trip"], res["net"]
    pred = res["pred"]
    # snapped positions of each GPS point on its predicted arc
    ok = pred >= 0
    snapped = np.full((len(pred), 2), np.nan)
    if ok.any():
        _, _, sc = net.point_arc(res["xy"][ok], pred[ok])
        q = net.A[pred[ok]] + net.unit[pred[ok]] * sc[:, None]
        snapped[ok] = net.unproject(q)
    gps = [[float(la), float(lo)] for lo, la in trip.track[:, :2]]
    body = dict(
        trip_id=tid,
        n_observations=int(len(trip.track)),
        gps_points=gps,
        predicted_arc_per_observation=[int(a) for a in pred],
        predicted_route_arcs=[int(a) for a in res["route"]],
        predicted_route_polyline=_polyline(net, trip.nodes, res["route"]),
        snapped_points=[[float(p[1]), float(p[0])] if np.isfinite(p[0]) else None for p in snapped],
        metrics=metrics,
        timing_seconds={k: round(v, 3) for k, v in res["timing"].items()},
    )
    m = metrics["hmm"] if metrics else {}
    c = metrics["classifier"] if metrics else {}
    log.info("trip=%s obs=%d | top1 classifier=%.4f hmm=%.4f | route F1 classifier=%.4f hmm=%.4f | continuity(relaxed) "
             "hmm=%.4f | time total=%.2fs (features %.2fs, clf %.2fs, hmm %.2fs)", tid, len(trip.track),
             c.get("top1_acc", float("nan")), m.get("top1_acc", float("nan")), c.get("route_f1", float("nan")),
             m.get("route_f1", float("nan")), m.get("continuity_relaxed", float("nan")), time.time() - t0,
             res["timing"]["features"], res["timing"]["classifier"], res["timing"]["hmm"])
    return body
