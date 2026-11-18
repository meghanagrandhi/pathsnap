"""Loading of the pre-built feature tables with the training-partition quality filter."""
import os

import numpy as np
import pandas as pd

from . import config as C


def load_partition(part, results_dir=C.RESULT_DIR, training=False):
    """training=True: drop trips failing label QC (train partition only) and observations whose reference
    arc is not among the candidates (they cannot be supervised). Student-test is never filtered."""
    df = pd.read_pickle(os.path.join(results_dir, f"features_{part}.pkl"))
    stats = pd.read_csv(os.path.join(results_dir, f"trip_stats_{part}.csv"), dtype={"trip": str})
    if training:
        ok = set(stats.loc[stats["usable"] == 1, "trip"])
        df = df[df["trip"].isin(ok) & (df["obs_hit"] == 1)]
    return df.reset_index(drop=True), stats


def xy_of(df):
    return df[C.FEATURES].to_numpy(dtype=np.float32), df["y"].to_numpy()


def obs_key(df):
    """Unique integer key per (trip, observation)."""
    codes = pd.factorize(df["trip"])[0].astype(np.int64)
    return codes * 10_000_000 + df["obs"].to_numpy(dtype=np.int64)


def top1_accuracy(df, prob, n_obs_total=None):
    """Observation-level Top-1 accuracy. Denominator = all observations (n_obs_total) if given, else
    the observations present in df."""
    key = obs_key(df)
    tmp = pd.DataFrame({"k": key, "p": prob, "hit": (df["arc"].to_numpy() == df["label_arc"].to_numpy())})
    tmp = tmp.sort_values(["k", "p"], ascending=[True, False])
    first = tmp.drop_duplicates("k")
    correct = int(first["hit"].sum())
    return correct / (n_obs_total if n_obs_total else len(first))
