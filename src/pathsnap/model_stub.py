"""Replaceable demo model interface."""

from __future__ import annotations


def predict_route(route_content: bytes) -> dict:
    """Return the route edge IDs from the uploaded .route file.

    The production model will return the same edge-ID format: one arc index per
    route step. This stub intentionally uses the reference route as its output.
    """
    route_arc_ids = [
        int(line)
        for line in route_content.decode("utf-8").splitlines()
        if line.strip()
    ]
    return {
        "route_arc_ids": route_arc_ids,
        "model": "demo_reference_route_v1",
        "is_demo": True,
    }
