# PathSnap

A local upload-first backend and map frontend for one complete trip. The backend runs the final XGBoost classifier and Viterbi HMM against uploaded trip data.

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

The UI displays the real OpenStreetMap base map, the `.route` reference graph, the uploaded `.track` user path, and the production prediction rendered from predicted edge IDs.

## API

- `GET /health`
- `POST /trip/analyze` with multipart fields `track`, `nodes`, `arcs`, and `route`

The production interface is `src/pathsnap/pipeline.py`; `src/pathsnap/model_stub.py` provides a compatibility wrapper.

## Test

```bash
pytest -q
```
