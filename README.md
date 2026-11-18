# PathSnap

A local upload-first backend and map frontend for one complete trip. A demo model returns the uploaded `.route` edge IDs until a production model is added.

## Run

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

Upload exactly these four whitespace-separated data files together:

- `.track`: longitude latitude timestamp
- `.nodes`: longitude latitude
- `.arcs`: from_node_id to_node_id
- `.route`: one reference arc ID per line

The UI displays the real OpenStreetMap base map, the `.route` reference graph, the uploaded `.track` user path, and the demo prediction rendered from predicted edge IDs.

## API

- `GET /health`
- `POST /trip/analyze` with multipart fields `track`, `nodes`, `arcs`, and `route`

The demo model interface is in `src/pathsnap/model_stub.py`. Replace `predict_route()` to connect a production model while preserving the edge-ID response contract.

## Test

```bash
pytest -q
```
