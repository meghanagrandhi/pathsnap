"""Compatibility interface for the production inference pipeline."""
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
    """Run production inference for parsed trip data."""
    if isinstance(trip_or_route, Mapping):
        return predict_parsed_trip(trip_or_route, get_model_bundle())
    raise TypeError("predict_route expects parsed trip data")
