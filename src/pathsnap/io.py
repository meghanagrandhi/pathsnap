"""Raw-file loading. Files are found by extension inside a trip folder (not by assuming names)."""
import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from_node, to_node
    route: Optional[np.ndarray]  # (M,) reference arc ids (None when not loaded)


def _find(folder, ext):
    hits = sorted(glob.glob(os.path.join(folder, "*." + ext)))
    return hits[0] if hits else None


def load_trip(folder: str, with_route: bool = True) -> Trip:
    """Load a trip folder. with_route=False guarantees reference data never enters the prediction path."""
    tid = os.path.basename(os.path.normpath(folder))
    track = np.loadtxt(_find(folder, "track"), ndmin=2)
    nodes = np.loadtxt(_find(folder, "nodes"), ndmin=2)
    arcs = np.loadtxt(_find(folder, "arcs"), dtype=np.int64, ndmin=2)
    route = None
    if with_route:
        rp = _find(folder, "route")
        if rp is not None and os.path.getsize(rp) > 0:
            route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    return Trip(tid, track, nodes, arcs, route)


def list_trip_dirs(root: str):
    """Trip folders under root (each containing a .track file)."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and _find(p, "track"):
            out.append(p)
    return out
