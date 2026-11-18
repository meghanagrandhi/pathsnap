"""Evaluation metrics.

Definitions (also in README):
 * Candidate-level classification (Macro F1, Precision, Recall, Log Loss): every (observation, candidate arc)
   row is a binary example, y=1 iff the arc is the reference arc. Macro F1 = mean of F1 for class 0 and 1.
 * Observation Top-1 accuracy: per observation, the highest-scoring candidate vs the reference arc.
   Observations whose reference arc is not among the candidates count as errors (denominator = all obs).
 * Candidate miss rate: share of observations whose reference arc is absent from the candidate set.
 * Route continuity: over consecutive observation pairs whose predicted arcs differ, the share that are
   network-connected. 'strict' = the next arc starts where the previous ends; 'relaxed' = the next arc is
   reachable within the distance the vehicle could plausibly cover (2*g+150 m).
 * Route F1: precision/recall/F1 of the set of arcs in the reconstructed route vs the reference route.
"""
import numpy as np
from .network import Network


def top1_choice(obs, prob, n_obs):
    """Row index of highest-probability candidate per observation (-1 if none)."""
    order = np.lexsort((-prob, obs))
    first = np.searchsorted(obs[order], np.arange(n_obs), side="left")
    has = np.bincount(obs, minlength=n_obs) > 0
    out = np.full(n_obs, -1, dtype=np.int64)
    out[has] = order[first[has]]
    return out


def obs_accuracy(chosen_rows, arc, labels):
    pred = np.where(chosen_rows >= 0, arc[np.maximum(chosen_rows, 0)], -1)
    return float(np.mean(pred == labels)), pred


def miss_rate(obs, arc, labels):
    hit = np.zeros(len(labels), dtype=bool)
    m = arc == labels[obs]
    hit[obs[m]] = True
    return float(1.0 - hit.mean())


def candidate_metrics(y, p, thr=0.5):
    from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
    pred = (p >= thr).astype(int)
    return dict(
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        precision=float(precision_score(y, pred, average="macro", zero_division=0)),
        recall=float(recall_score(y, pred, average="macro", zero_division=0)),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    )


def route_continuity(net: Network, pred_arcs, xy, dt, dist_factor=2.0, slack=150.0):
    strict_ok = relaxed_ok = total = 0
    for i in range(1, len(pred_arcs)):
        a, b = int(pred_arcs[i - 1]), int(pred_arcs[i])
        if a < 0 or b < 0 or a == b:
            continue
        total += 1
        if net.to[a] == net.frm[b]:
            strict_ok += 1
            relaxed_ok += 1
            continue
        g = float(np.hypot(*(xy[i] - xy[i - 1])))
        limit = dist_factor * g + slack
        d, _ = net.shortest(int(net.to[a]), limit)
        v = d.get(int(net.frm[b]))
        if v is not None and v <= limit:
            relaxed_ok += 1
    if total == 0:
        return 1.0, 1.0, 0
    return strict_ok / total, relaxed_ok / total, total


def route_prf(pred_route, ref_route):
    P, R = set(int(x) for x in pred_route), set(int(x) for x in ref_route)
    inter = len(P & R)
    prec = inter / max(len(P), 1)
    rec = inter / max(len(R), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
