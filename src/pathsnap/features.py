"""Candidate generation + per-(observation, candidate-arc) feature engineering.

Features are purely geometric / contextual so they transfer across trips (each trip has its own road
network, so arc ids are meaningless as features). No reference-route information is used here.
"""
import numpy as np
import pandas as pd

from . import config as C
from .io import Trip
from .network import Network


def motion_features(xy: np.ndarray, t: np.ndarray, half: int = C.HEADING_HALF_WINDOW):
    """Heading (rad), speed (m/s) and path length (m) over a centred window; dt to prev/next obs."""
    N = len(xy)
    i = np.arange(N)
    j0 = np.clip(i - half, 0, N - 1)
    j1 = np.clip(i + half, 0, N - 1)
    v = xy[j1] - xy[j0]
    dtw = np.maximum(t[j1] - t[j0], 1e-3)
    dist = np.hypot(v[:, 0], v[:, 1])
    speed = dist / dtw
    heading = np.arctan2(v[:, 1], v[:, 0])
    valid = (speed > C.MIN_SPEED_FOR_HEADING) & (j1 > j0)
    dt_prev = np.where(i > 0, t - t[np.clip(i - 1, 0, N - 1)], 1.0)
    dt_next = np.where(i < N - 1, t[np.clip(i + 1, 0, N - 1)] - t, 1.0)
    return heading, speed, valid, dt_prev, dt_next


def build_candidate_table(trip: Trip, net: Network = None, radius=C.CAND_RADIUS_M, max_k=C.MAX_CANDIDATES,
                          min_k=C.MIN_CANDIDATES) -> pd.DataFrame:
    """One row per (observation, candidate arc) with all model features. Sorted by obs then distance."""
    if net is None:
        net = Network(trip.nodes, trip.arcs, C.SAMPLE_STEP_M)
    xy = net.project(trip.track[:, :2])
    t = trip.track[:, 2]
    N = len(xy)
    o, a, d0 = net.candidates(xy, radius, max_k, min_k)
    heading, speed, hvalid, dt_prev, dt_next = motion_features(xy, t)

    df = pd.DataFrame({"obs": o, "arc": a, "d": d0})
    g = df.groupby("obs")["d"]
    df["n_cand"] = g.transform("size").astype(np.float32)
    df["d_rank"] = g.rank(method="first").astype(np.float32)
    dmin = g.transform("min")
    df["d_gap"] = df["d"] - dmin
    df["d_ratio"] = df["d"] / (dmin + 1.0)

    _, s, sc = net.point_arc(xy[o], a)
    L = net.length[a]
    df["frac"] = sc / np.maximum(L, 1e-6)
    df["along"] = sc
    df["arc_len"] = L
    df["remain"] = L - sc

    # heading agreement between motion direction and arc direction
    dh = heading[o] - net.bearing[a]
    cos_dh = np.cos(dh)
    hv = hvalid[o]
    df["cos_dh"] = np.where(hv, cos_dh, 0.0)
    df["abs_dh"] = np.where(hv, np.abs(np.arctan2(np.sin(dh), cos_dh)), np.pi / 2)
    df["heading_valid"] = hv.astype(np.float32)
    df["speed"] = speed[o]

    # temporal context: distance of neighbouring observations to the SAME arc
    def nb(k):
        idx = np.clip(o + k, 0, N - 1)
        dd, ss, _ = net.point_arc(xy[idx], a)
        return dd, ss

    d_m1, _ = nb(-1)
    d_p1, _ = nb(1)
    d_m2, s_m2 = nb(-2)
    d_p2, s_p2 = nb(2)
    df["d_m1"], df["d_p1"], df["d_m2"], df["d_p2"] = d_m1, d_p1, d_m2, d_p2
    df["d_win_mean"] = (df["d"] + d_m1 + d_p1 + d_m2 + d_p2) / 5.0
    wg = df.groupby("obs")["d_win_mean"]
    df["d_win_gap"] = df["d_win_mean"] - wg.transform("min")
    df["d_win_rank"] = wg.rank(method="first").astype(np.float32)

    # signed progress of the vehicle along the arc direction over the +-2 window
    prog = s_p2 - s_m2
    i0 = np.clip(o - 2, 0, N - 1)
    i1 = np.clip(o + 2, 0, N - 1)
    path = np.hypot(*(xy[i1] - xy[i0]).T)
    df["prog"] = prog
    df["prog_ratio"] = prog / (path + 1.0)

    df["has_rev"] = net.has_rev[a]
    df["deg_to_out"] = net.out_deg[net.to[a]].astype(np.float32)
    df["deg_from_in"] = net.in_deg[net.frm[a]].astype(np.float32)
    df["dt_prev"] = dt_prev[o]
    df["dt_next"] = dt_next[o]
    # position along the arc is needed by the HMM; `along` is stored above
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[C.FEATURES].to_numpy(dtype=np.float32)
