"""HMM / Viterbi route reconstruction on top of classifier candidate probabilities.

Hidden state at time t : one candidate road arc of observation t (the candidate-to-arc correspondence is
                         preserved: state k at time t <-> row k of that observation's candidate table).
Emission               : log of the classifier probability, renormalised over the observation's
                         candidates and scaled by tau  ->  e_t(a) = tau * log(p_t(a) / sum_b p_t(b)).
Transition             : Newson-Krumm style, comparing great-circle distance g between consecutive GPS
                         points with the network route distance rd between the two candidate positions:
                         log T = -|g - rd| / (beta * max(1, dt)).
Unreachable / invalid  : if b is not reachable from a within the search limit, log T = -unreachable_penalty
                         (finite, so the chain can still continue through a disconnected jump).
"""
import numpy as np

from . import config as C
from .network import Network


def _route_dist(net: Network, a, sa, b, sb, limit, backward_pen_m=5.0):
    """Network travel distance from position sa on arc a to position sb on arc b (inf if unreachable)."""
    if a == b:
        if sb >= sa - backward_pen_m:
            return max(sb - sa, 0.0), False
        return sa - sb, True  # moving backwards along the arc
    base = net.length[a] - sa
    na, nb = net.to[a], net.frm[b]
    if na == nb:
        return base + sb, False
    dist, _ = net.shortest(int(na), limit)
    v = dist.get(int(nb))
    if v is None:
        return np.inf, False
    return base + v + sb, False


def viterbi(net: Network, xy, t, obs, arc, along, prob, cfg=None):
    """Decode. obs/arc/along/prob are candidate-table columns (sorted by obs).
    Returns chosen row index (into the table) for each observation (-1 if it had no candidates)."""
    cfg = {**C.HMM_DEFAULT, **(cfg or {})}
    N = len(xy)
    starts = np.searchsorted(obs, np.arange(N + 1), side="left")
    chosen = np.full(N, -1, dtype=np.int64)

    # emission log-probabilities
    lp = np.empty(len(obs))
    for i in range(N):
        s, e = starts[i], starts[i + 1]
        if e > s:
            p = np.maximum(prob[s:e], 1e-9)
            lp[s:e] = cfg["tau"] * np.log(p / p.sum())

    prev_rows, prev_score, prev_t = None, None, None
    back = {}
    for i in range(N):
        s, e = starts[i], starts[i + 1]
        if e == s:
            continue  # no candidates: skip observation, chain continues from last valid layer
        rows = np.arange(s, e)
        if prev_rows is None:
            score = lp[s:e].copy()
            back[i] = (None, None)
        else:
            dt = max(t[i] - prev_t, 1.0)
            g = float(np.hypot(*(xy[i] - xy[prev_i])))
            limit = cfg["dist_factor"] * g + cfg["dist_slack_m"]
            beta = cfg["beta"] * (max(1.0, dt) if cfg["gap_scale"] else 1.0)
            trans = np.empty((len(prev_rows), len(rows)))
            for m, pr in enumerate(prev_rows):
                for n, r in enumerate(rows):
                    rd, backward = _route_dist(net, int(arc[pr]), float(along[pr]), int(arc[r]), float(along[r]), limit)
                    if np.isinf(rd):
                        trans[m, n] = -cfg["unreachable_penalty"]
                    else:
                        trans[m, n] = -abs(g - rd) / beta - (cfg["backward_penalty"] if backward else 0.0)
            cand = prev_score[:, None] + trans
            best = np.argmax(cand, axis=0)
            score = cand[best, np.arange(len(rows))] + lp[s:e]
            back[i] = (prev_i, best)
        prev_rows, prev_score, prev_t, prev_i = rows, score, t[i], i
    if prev_rows is None:
        return chosen
    # backtrack
    k = int(np.argmax(prev_score))
    i = prev_i
    while i is not None:
        chosen[i] = starts[i] + k
        pi, best = back[i]
        if pi is None:
            break
        k = int(best[k])
        i = pi
    return chosen


def fill_route(net: Network, pred_arcs, max_fill_m=1500.0):
    """Turn per-observation arcs into a connected arc sequence (consecutive duplicates removed; gaps
    between non-adjacent arcs bridged with the shortest network path when one exists)."""
    seq = []
    for a in pred_arcs:
        if a < 0:
            continue
        a = int(a)
        if not seq:
            seq.append(a)
            continue
        last = seq[-1]
        if a == last:
            continue
        if net.to[last] != net.frm[a]:
            mid = net.path_arcs(int(net.to[last]), int(net.frm[a]), max_fill_m)
            if mid:
                seq.extend(int(m) for m in mid)
        seq.append(a)
    return seq
