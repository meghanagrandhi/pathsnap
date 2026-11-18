# PathSnap - distributed map-matching & route reconstruction

Noisy 1 Hz GPS fixes -> road-network arcs. Pipeline:

1. **Candidate generation** - per GPS fix, all arcs within 50 m (max 12 nearest; fallback to nearest arcs if fewer than 3).
2. **Candidate classifier** - every (fix, candidate arc) row is scored with geometric/contextual features
   (distance, heading agreement, along-arc position, distance of neighbouring fixes to the same arc, progress direction, ...).
   Arc ids are *not* features (each trip has its own network).
3. **HMM / Viterbi** - hidden state = candidate arc of each fix; emission = classifier probability; transition = |GPS distance - network route distance|.
4. **FastAPI** server (Laptop 1) + **static HTML/Leaflet UI** (Laptop 2).

```
src/pathsnap/   shared package (io, network, features, labels, hmm, metrics, models, pipeline, evaluate)
scripts/        01_eda, 02_build_features, 03_train_compare, 04_bayes_opt, 05_hmm_tune_eval, 06_folium_maps, 07_plots
api/main.py     FastAPI backend          ui/index.html   evaluator UI
artifacts/      model.txt (LightGBM), model_meta.json, hmm_config.json   (created by scripts 04/05)
results/        CSV results, experiment logs     figures/   plots + Folium HTML maps
notebooks/colab_run_all.ipynb   runs the whole experiment on Colab
```

## Definitions (used consistently everywhere)
* **Reference label per fix**: `.route` is only an ordered arc list, so each fix is aligned to the route with a monotone
  dynamic-programming alignment (`labels.py`) minimising capped fix-to-arc distance. The same function produces training labels,
  student-test scores and the API's server-side score. Reference routes are treated as *reference*, not verified truth.
* **Macro F1 / Precision / Recall / Log loss**: candidate-row binary classification (y=1 iff row's arc is the reference arc); macro = mean over both classes, threshold 0.5.
* **Top-1 accuracy**: per fix, highest-scoring candidate == reference arc; fixes whose reference arc is not a candidate count as wrong.
* **Candidate miss rate**: fixes whose reference arc is absent from the candidate set.
* **Route continuity**: of consecutive fixes with different predicted arcs, share that are network-connected
  (*strict*: next arc starts where previous ends; *relaxed*: reachable within 2*g+150 m).
* **Route F1**: precision/recall/F1 of the arc set of the reconstructed (gap-filled) route vs the reference route.
* **CV**: 5-fold `GroupKFold` grouped by trip on the 80 training trips (no trip split across folds).

## Data-quality decisions (see `results/trip_stats_train.csv`)
Some training trips have a reference route that does not describe the recorded drive (mean fix-to-reference distance far above GPS noise;
e.g. trips 2, 33, 43, 54, 61, 66, 72). Trips with mean label distance > 15 m or candidate-miss rate > 5 % (`config.QC_*`) are excluded from
fitting/CV, and individual fixes whose reference arc is outside the candidate set are not used as supervision. Student-test trips are never filtered.
Evaluator-test data is never used for any choice.

## Run the experiments (Colab recommended)
Open `notebooks/colab_run_all.ipynb`, or locally:
```
pip install -r requirements.txt
python scripts/01_eda.py              --data-root <dir with train/ and student_test/>
python scripts/02_build_features.py   --data-root <...>
python scripts/03_train_compare.py    --data-root <...>          # 7 models + nearest-arc baseline
python scripts/04_bayes_opt.py        --data-root <...> --trials 30   # Optuna TPE; exports artifacts/model.txt
python scripts/05_hmm_tune_eval.py    --data-root <...> --model LightGBM_tuned --oof oof_tuned.pkl --student student_tuned.pkl
python scripts/07_plots.py ; python scripts/06_folium_maps.py --data-root <...> --trip <id>
```
Model selection is made from `results/model_comparison.csv`; script 04 tunes LightGBM - if your comparison selects a different model,
change `04_bayes_opt.py` accordingly and document why. Decoder parameters (`05`) are tuned on out-of-fold *training* predictions only.
Note: scikit-learn tree models need a working `sklearn.tree` DLL (some locked-down Windows machines block it); the served model is native LightGBM and needs no sklearn.

## Deployment (two laptops)
**Laptop 1 (API)** - copy repo + the folder of the 10 reserved trips (each trip folder with `.track .nodes .arcs .route`):
```
pip install -r requirements.txt
set PATHSNAP_TRIPS_DIR=D:\evaluator_trips        (PowerShell: $env:PATHSNAP_TRIPS_DIR="D:\evaluator_trips")
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Allow TCP 8000 in Windows Firewall; find the IP with `ipconfig`. Every request prints trip id, metrics and timings to this console (show it to evaluators).
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Reference routes stay server-side; responses contain only predictions and scores.

**Laptop 2 (UI)**:
```
python -m http.server 8080 --directory ui
```
open `http://localhost:8080`, enter `http://<laptop1-ip>:8000`, click *Load trips*, then a trip. (Leaflet/OSM tiles load from the internet; for offline
use copy Leaflet 1.9.4 into `ui/vendor/leaflet/` - route/GPS layers draw without tiles.)

## AI-assistance disclosure
This code base was drafted with an AI coding assistant (Claude Code). All numbers in reports must come from running the scripts; the
team is responsible for understanding and being able to explain every component.
