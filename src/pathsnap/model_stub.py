"""Compatibility interface for production and legacy route-only callers."""
from __future__ import annotations

from typing import Any, Mapping

from .pipeline import ModelBundle, predict_parsed_trip

_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle


def predict_route(trip_or_route: Any) -> dict:
    """Predict parsed trip data in production; route bytes remain a legacy compatibility input."""
    if isinstance(trip_or_route, Mapping):
        return predict_parsed_trip(trip_or_route, get_model_bundle())

    if isinstance(trip_or_route, bytes):
        route_arc_ids = [
            int(line)
            for line in trip_or_route.decode("utf-8").splitlines()
            if line.strip()
        ]
        return {
            "route_arc_ids": route_arc_ids,
            "model": "demo_reference_route_v1",
            "is_demo": True,
        }

    raise TypeError("predict_route expects parsed trip data or legacy route bytes")
