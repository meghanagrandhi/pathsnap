"""Strict parser for one uploaded PathSnap trip."""

from __future__ import annotations

import math
from pathlib import PurePath


class TripValidationError(ValueError):
    pass


def route_segments_for_ids(nodes: list[list[float]], arcs: list[list[int]], arc_ids: list[int]):
    """Convert arc IDs into Leaflet [latitude, longitude] line segments."""
    segments = []
    for arc_id in arc_ids:
        from_id, to_id = arcs[arc_id]
        from_lon, from_lat = nodes[from_id]
        to_lon, to_lat = nodes[to_id]
        segments.append([[from_lat, from_lon], [to_lat, to_lon]])
    return segments


def _text(filename: str, content: bytes) -> str:
    if not content:
        raise TripValidationError(f"{filename} is empty")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TripValidationError(f"{filename} must be UTF-8 text") from exc


def _rows(filename: str, content: bytes, columns: int):
    text = _text(filename, content)
    rows = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != columns:
            raise TripValidationError(
                f"{filename} line {line_number}: expected {columns} whitespace-separated columns, got {len(parts)}"
            )
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise TripValidationError(f"{filename} line {line_number}: all values must be numbers") from exc
    if not rows:
        raise TripValidationError(f"{filename} contains no data rows")
    return rows


def parse_trip(files: dict[str, tuple[str, bytes]]) -> dict:
    required = {"track", "nodes", "arcs", "route"}
    if set(files) != required:
        missing = sorted(required - set(files))
        extra = sorted(set(files) - required)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise TripValidationError("; ".join(details))

    track_rows = _rows(files["track"][0], files["track"][1], 3)
    node_rows = _rows(files["nodes"][0], files["nodes"][1], 2)
    arc_rows = _rows(files["arcs"][0], files["arcs"][1], 2)
    route_rows = _rows(files["route"][0], files["route"][1], 1)

    for row_number, (longitude, latitude, _timestamp) in enumerate(track_rows, 1):
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            raise TripValidationError(f"{files['track'][0]} line {row_number}: invalid coordinates")
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise TripValidationError(f"{files['track'][0]} line {row_number}: coordinates outside valid bounds")

    nodes = []
    for row_number, (longitude, latitude) in enumerate(node_rows, 1):
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            raise TripValidationError(f"{files['nodes'][0]} line {row_number}: invalid coordinates")
        nodes.append([longitude, latitude])

    arcs = []
    for row_number, (from_raw, to_raw) in enumerate(arc_rows, 1):
        if not (from_raw.is_integer() and to_raw.is_integer()):
            raise TripValidationError(f"{files['arcs'][0]} line {row_number}: node IDs must be integers")
        from_id, to_id = int(from_raw), int(to_raw)
        if not (0 <= from_id < len(nodes) and 0 <= to_id < len(nodes)):
            raise TripValidationError(f"{files['arcs'][0]} line {row_number}: arc node ID outside nodes range")
        arcs.append([from_id, to_id])

    route = []
    for row_number, (arc_raw,) in enumerate(route_rows, 1):
        if not arc_raw.is_integer():
            raise TripValidationError(f"{files['route'][0]} line {row_number}: arc ID must be an integer")
        arc_id = int(arc_raw)
        if not (0 <= arc_id < len(arcs)):
            raise TripValidationError(f"{files['route'][0]} line {row_number}: route arc ID outside arcs range")
        route.append(arc_id)

    gps_points = [[longitude, latitude] for longitude, latitude, _timestamp in track_rows]
    gps_latlng = [[latitude, longitude] for longitude, latitude in gps_points]

    road_segments = []
    for from_id, to_id in arcs:
        from_lon, from_lat = nodes[from_id]
        to_lon, to_lat = nodes[to_id]
        road_segments.append([[from_lat, from_lon], [to_lat, to_lon]])

    route_segments = route_segments_for_ids(nodes, arcs, route)

    return {
        "gps_points": gps_latlng,
        "gps_longitude_latitude": gps_points,
        "timestamps": [timestamp for _longitude, _latitude, timestamp in track_rows],
        "road_segments": road_segments,
        "route_segments": route_segments,
        "route_arc_ids": route,
        "nodes": nodes,
        "arcs": arcs,
        "node_count": len(nodes),
        "arc_count": len(arcs),
    }
