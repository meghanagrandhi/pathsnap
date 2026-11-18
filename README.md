# PathSnap

A local PathSnap backend and map frontend. The backend runs the final XGBoost classifier and Viterbi HMM against either a valid backend test dataset ID or uploaded trip data.

## Run

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The `0.0.0.0` bind makes the frontend available to other laptops on the same
network. On this laptop, open <http://10.30.204.59:8000> from another device
(or replace the IP with the host laptop's current LAN address). Allow TCP port
`8000` through the host firewall if the other laptop cannot connect.

You can use either input mode:

1. Enter one of the ten valid test dataset IDs:

   `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`

2. Upload exactly these four whitespace-separated data files together:

- `.track`: longitude latitude timestamp
- `.nodes`: longitude latitude
- `.arcs`: from_node_id to_node_id
- `.route`: one reference arc ID per line

The UI displays the real OpenStreetMap base map, the `.route` reference graph, the uploaded `.track` user path, and the production prediction rendered from predicted edge IDs.

Predictions are shown exactly as returned by the XGBoost + Viterbi pipeline.
The application does not substitute the reference route, nearest road, or a
corrected path when a prediction is poor. Candidate generation may select
nearest arcs when no arc is within the configured search radius; this is part
of the model's defined input pipeline, not a prediction fallback.

## API

- `GET /health`
- `GET /test-datasets`
- `POST /trip/analyze-by-id` with JSON body `{"dataset_id":"00000000"}`
- `POST /trip/analyze` with multipart fields `track`, `nodes`, `arcs`, and `route`

The production interface is `src/pathsnap/pipeline.py`; `src/pathsnap/model_stub.py` provides a compatibility wrapper.

## Test

```bash
pytest -q
```
