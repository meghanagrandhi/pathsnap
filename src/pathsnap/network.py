"""Road network: local metric projection, arc geometry, spatial index, bounded shortest paths."""
import heapq
import math

import numpy as np
from scipy.spatial import cKDTree

EARTH_R = 6371008.8


class Network:
    def __init__(self, nodes_ll: np.ndarray, arcs: np.ndarray, sample_step: float = 15.0):
        # Local equirectangular projection centred on the network. Trip extents are <= ~50 km, so
        # distortion is well below GPS noise; degrees are never used as distances.
        self.lon0 = float(nodes_ll[:, 0].mean())
        self.lat0 = float(nodes_ll[:, 1].mean())
        self.xy = self.project(nodes_ll)
        self.n_nodes = len(nodes_ll)
        self.frm = arcs[:, 0].astype(np.int64)
        self.to = arcs[:, 1].astype(np.int64)
        self.n_arcs = len(arcs)
        self.A = self.xy[self.frm]
        self.B = self.xy[self.to]
        vec = self.B - self.A
        self.length = np.hypot(vec[:, 0], vec[:, 1])
        self.unit = vec / np.maximum(self.length, 1e-6)[:, None]
        self.bearing = np.arctan2(vec[:, 1], vec[:, 0])
        self.out_deg = np.bincount(self.frm, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.to, minlength=self.n_nodes)
        key = self.frm * self.n_nodes + self.to
        rkey = self.to * self.n_nodes + self.frm
        self.has_rev = np.isin(rkey, key).astype(np.float32)
        # CSR adjacency (arcs sorted by from-node) for Dijkstra
        order = np.argsort(self.frm, kind="stable")
        ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(self.out_deg, out=ptr[1:])
        self._order = order.tolist()
        self._ptr = ptr.tolist()
        self._to = self.to.tolist()
        self._len = self.length.tolist()
        self._sp_cache = {}
        self.sample_step = sample_step
        self._tree = None

    # ---- projection -------------------------------------------------
    def project(self, ll: np.ndarray) -> np.ndarray:
        x = np.radians(ll[:, 0] - self.lon0) * EARTH_R * math.cos(math.radians(self.lat0))
        y = np.radians(ll[:, 1] - self.lat0) * EARTH_R
        return np.column_stack([x, y])

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        lon = self.lon0 + np.degrees(xy[:, 0] / (EARTH_R * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + np.degrees(xy[:, 1] / EARTH_R)
        return np.column_stack([lon, lat])

    # ---- geometry ---------------------------------------------------
    def point_arc(self, p: np.ndarray, arc: np.ndarray):
        """Distance from points p (n,2) to arcs `arc` (n,), plus unclipped along-arc coordinate s (m)."""
        A = self.A[arc]
        u = self.unit[arc]
        s = ((p - A) * u).sum(1)
        sc = np.clip(s, 0.0, self.length[arc])
        q = A + u * sc[:, None]
        d = np.hypot(p[:, 0] - q[:, 0], p[:, 1] - q[:, 1])
        return d, s, sc

    # ---- spatial index ---------------------------------------------
    def _build_tree(self):
        counts = np.maximum(np.ceil(self.length / self.sample_step), 1).astype(np.int64) + 1
        arc_ids = np.repeat(np.arange(self.n_arcs), counts)
        starts = np.cumsum(counts) - counts
        k = np.arange(counts.sum()) - np.repeat(starts, counts)
        t = k / np.repeat(counts - 1, counts)
        pts = self.A[arc_ids] + (self.B[arc_ids] - self.A[arc_ids]) * t[:, None]
        self._tree = cKDTree(pts)
        self._tree_arc = arc_ids

    def candidates(self, obs_xy: np.ndarray, radius: float, max_k: int, min_k: int):
        """Return (obs_idx, arc_id, dist) rows, sorted by obs then distance."""
        if self._tree is None:
            self._build_tree()
        hits = self._tree.query_ball_point(obs_xy, r=radius + self.sample_step / 2.0)
        fb_d, fb_i = self._tree.query(obs_xy, k=min(40, len(self._tree_arc)))
        fb_i = np.atleast_2d(fb_i)
        o_list, a_list = [], []
        for i, h in enumerate(hits):
            arcs = np.unique(self._tree_arc[h]) if len(h) else np.empty(0, dtype=np.int64)
            o_list.append(np.full(len(arcs), i, dtype=np.int64))
            a_list.append(arcs)
        o = np.concatenate(o_list)
        a = np.concatenate(a_list)
        d, _, _ = self.point_arc(obs_xy[o], a)
        keep = d <= radius
        o, a, d = o[keep], a[keep], d[keep]
        # fallback for observations with too few candidates: nearest arcs regardless of radius
        cnt = np.bincount(o, minlength=len(obs_xy))
        need = np.where(cnt < min_k)[0]
        if len(need):
            fo, fa = [], []
            for i in need:
                arcs = np.unique(self._tree_arc[fb_i[i]])
                fo.append(np.full(len(arcs), i, dtype=np.int64))
                fa.append(arcs)
            fo = np.concatenate(fo)
            fa = np.concatenate(fa)
            fd, _, _ = self.point_arc(obs_xy[fo], fa)
            # drop earlier rows of those observations and replace with the fallback set
            mask = ~np.isin(o, need)
            o = np.concatenate([o[mask], fo])
            a = np.concatenate([a[mask], fa])
            d = np.concatenate([d[mask], fd])
        order = np.lexsort((d, o))
        o, a, d = o[order], a[order], d[order]
        # keep top max_k per observation (rows sorted by distance within obs)
        first = np.searchsorted(o, o, side="left")
        rank = np.arange(len(o)) - first
        sel = rank < np.where(np.isin(o, need), max(min_k, 5), max_k)
        return o[sel], a[sel], d[sel]

    # ---- bounded shortest paths ---------------------------------------
    def shortest(self, src: int, limit: float):
        """Dijkstra from node src, exploring up to `limit` metres. Returns (dist dict, pred-arc dict)."""
        limit = math.ceil(limit / 250.0) * 250.0
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        ptr, order, to, ln = self._ptr, self._order, self._to, self._len
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, 1e18):
                continue
            for k in range(ptr[u], ptr[u + 1]):
                a = order[k]
                v = to[a]
                nd = d + ln[a]
                if nd <= limit and nd < dist.get(v, 1e18):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (limit, dist, pred)
        return dist, pred

    def path_arcs(self, src_node: int, dst_node: int, limit: float):
        """Arc ids on the shortest path src_node -> dst_node (within limit); None if unreachable."""
        if src_node == dst_node:
            return []
        dist, pred = self.shortest(src_node, limit)
        if dst_node not in dist:
            return None
        out = []
        v = dst_node
        while v != src_node:
            a = pred[v]
            out.append(a)
            v = self.frm[a]
        return out[::-1]
