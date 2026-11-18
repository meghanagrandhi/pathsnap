"""Per-observation reference labels.

The .route file is only an ordered list of arcs; no row-to-observation mapping is supplied. We derive
one with a monotone alignment: each GPS observation i is assigned a route index j(i), with j
non-decreasing, minimising the sum of (capped) distances from observation to its assigned route arc.
This single function is used for training labels, student-test scoring and the API's server-side
evaluation, so every number in the project uses the same definition of 'reference arc'.
"""
import numpy as np

from .network import Network

CAP_M = 100.0


def align_obs_to_route(net: Network, obs_xy: np.ndarray, route: np.ndarray):
    """Return (label_arc (N,), route_index (N,), dist_to_label (N,))."""
    N, M = len(obs_xy), len(route)
    # D[i, j] = distance from obs i to route arc j
    D = np.empty((N, M), dtype=np.float32)
    A = net.A[route]
    u = net.unit[route]
    L = net.length[route]
    for i in range(N):
        s = ((obs_xy[i] - A) * u).sum(1)
        sc = np.clip(s, 0.0, L)
        q = A + u * sc[:, None]
        D[i] = np.hypot(obs_xy[i, 0] - q[:, 0], obs_xy[i, 1] - q[:, 1])
    C = np.minimum(D, CAP_M)
    idx = np.arange(M)
    back = np.zeros((N, M), dtype=np.int32)
    dp = C[0].copy()
    for i in range(1, N):
        cm = np.minimum.accumulate(dp)
        arg = np.maximum.accumulate(np.where(dp <= cm, idx, 0))
        back[i] = arg
        dp = C[i] + cm
    j = int(np.argmin(dp))
    ridx = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        ridx[i] = j
        j = back[i, j]
    labels = route[ridx]
    return labels, ridx, D[np.arange(N), ridx]
