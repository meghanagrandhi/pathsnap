"""PathSnap upload API. It accepts one complete trip and runs a replaceable demo model."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pathsnap.model_stub import predict_route  # noqa: E402
from src.pathsnap.trip import TripValidationError, parse_trip, route_segments_for_ids  # noqa: E402

app = FastAPI(title="PathSnap", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "demo_reference_route_v1"}


@app.get("/", include_in_schema=False)
def frontend():
    index = ROOT / "ui" / "index.html"
    return FileResponse(index)


@app.post("/trip/analyze")
async def analyze_trip(
    track: UploadFile = File(...),
    nodes: UploadFile = File(...),
    arcs: UploadFile = File(...),
    route: UploadFile = File(...),
):
    uploads = {"track": track, "nodes": nodes, "arcs": arcs, "route": route}
    for kind, upload in uploads.items():
        if Path(upload.filename or "").suffix.lower() != f".{kind}":
            raise HTTPException(status_code=422, detail=f"{kind} upload must have a .{kind} filename")

    files = {}
    try:
        for kind, upload in uploads.items():
            content = await upload.read()
            files[kind] = (upload.filename or f"trip.{kind}", content)
        trip = parse_trip(files)
    except TripValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = predict_route(files["route"][1])
    predicted_route_segments = route_segments_for_ids(
        trip["nodes"], trip["arcs"], prediction["route_arc_ids"]
    )

    return {
        "gps_points": trip["gps_points"],
        "timestamps": trip["timestamps"],
        "road_segments": trip["road_segments"],
        "route_segments": trip["route_segments"],
        "route_arc_ids": trip["route_arc_ids"],
        "predicted_route_arc_ids": prediction["route_arc_ids"],
        "predicted_route_segments": predicted_route_segments,
        "model": prediction["model"],
        "is_demo": prediction["is_demo"],
        "counts": {
            "gps_observations": len(trip["gps_points"]),
            "nodes": trip["node_count"],
            "arcs": trip["arc_count"],
            "reference_route_arcs": len(trip["route_arc_ids"]),
        },
    }
