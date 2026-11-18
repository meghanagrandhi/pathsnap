from src.pathsnap.model_stub import predict_route
from src.pathsnap.trip import TripValidationError, parse_trip


def make_files():
    return {
        "track": ("trip.track", b"10 20 1\n10.001 20.001 2\n"),
        "nodes": ("trip.nodes", b"10 20\n10.001 20.001\n"),
        "arcs": ("trip.arcs", b"0 1\n"),
        "route": ("trip.route", b"0\n"),
    }


def test_parse_trip_builds_map_layers():
    trip = parse_trip(make_files())
    assert trip["node_count"] == 2
    assert trip["arc_count"] == 1
    assert trip["route_arc_ids"] == [0]
    assert trip["gps_points"] == [[20.0, 10.0], [20.001, 10.001]]


def test_rejects_arc_outside_nodes():
    files = make_files()
    files["arcs"] = ("trip.arcs", b"0 9\n")
    try:
        parse_trip(files)
    except TripValidationError as exc:
        assert "outside nodes range" in str(exc)
    else:
        raise AssertionError("invalid arc should be rejected")


def test_production_model_loads_and_predicts_synthetic_trip():
    from src.pathsnap.pipeline import ModelBundle, predict_parsed_trip

    bundle = ModelBundle()
    parsed = {
        "gps_points": [[20.0, 10.0], [20.001, 10.001]],
        "gps_longitude_latitude": [[10.0, 20.0], [10.001, 20.001]],
        "timestamps": [1.0, 2.0],
        "nodes": [[10.0, 20.0], [10.001, 20.001]],
        "arcs": [[0, 1]],
        "route_arc_ids": [0],
        "road_segments": [[[20.0, 10.0], [20.001, 10.001]]],
        "route_segments": [[[20.0, 10.0], [20.001, 10.001]]],
    }
    prediction = predict_parsed_trip(parsed, bundle)
    assert prediction["route_arc_ids"]
    assert prediction["model"] == "xgboost_viterbi_v1"
    assert prediction["is_demo"] is False
