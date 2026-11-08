# Raga Recommender Backend

Local FastAPI backend for the Raga Recommender System.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend uses the root-level `raga_sthayi_bhava_dataset_enriched.json` and
`best_navarasa_model.pth` by default. Override them with `RAGA_DATA_PATH` and
`RASA_MODEL_PATH` if needed.

