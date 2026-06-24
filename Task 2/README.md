
# IMDb Movie Rating Predictor — Task 2

This repository contains a simple Streamlit app that predicts IMDb-style movie ratings using a trained Random Forest regression model. The app was built as a compact, beginner-friendly example to demonstrate loading model artifacts, encoding categorical inputs, and making predictions in a small web UI.

Contents (inside `Task 2/`):

- `Task 2/app.py` — Streamlit application. Simple, step-by-step UI that accepts movie details and shows a predicted rating.
- `Task 2/movie_model.pkl` — Pickled scikit-learn RandomForestRegressor trained on the provided dataset.
- `Task 2/target_encoders.pkl` — Pickled dictionary mapping categorical values (Genre, Director, Actor 1/2/3) to mean target ratings (used as simple target-encoding).
- `Task 2/requirements.txt` — Python dependencies to run the app.
- `Task 2/IMDb Movies India.csv` — Source dataset used to train the model (kept here for reference).

Overview
--------

The app loads `movie_model.pkl` and `target_encoders.pkl` at startup. For categorical inputs (Genre, Director, Actor 1/2/3) the app looks up the mean-encoded value from `target_encoders.pkl`. If a provided categorical value was not seen in training, the app uses a default fallback value of `5.8` (mid-range rating) and shows a small warning.

Model details
-------------

- Model type: `RandomForestRegressor` from scikit-learn.
- Training dataset: `IMDb Movies India.csv` (cleaned input features include Year, Duration, Votes, Genre, Director, Actor 1/2/3).
- Encoders: For each categorical column we computed the mean rating per category on the training set and saved those mappings in `target_encoders.pkl`. During inference these means are used as numeric substitutions.
- Fallback: unseen categorical values map to `5.8` to avoid exceptions during prediction.
- Output bounds: predicted rating is clipped to the range `[1.0, 10.0]` before display.

How to run locally
-------------------

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS / Linux
```

2. Install dependencies:

```bash
python -m pip install -r "Task 2/requirements.txt"
```

3. Run the Streamlit app:

```bash
python -m streamlit run "Task 2/app.py"
```

4. Open the displayed Local URL (usually `http://localhost:8501`).

Notes about committing model files
---------------------------------

- This repo currently contains `movie_model.pkl` and the training CSV. If you prefer a small repo without large binaries, consider removing these files from the repository and adding them to GitHub Releases, an S3 bucket, or another file host, then update `README.md` with download instructions.
- To stop tracking binary files moving forward, add a `.gitignore` with entries like:

```
*.pkl
__pycache__/
.venv/

```

And (if you want them removed from history) use tools like `git filter-repo` or BFG to rewrite history — contact me if you want help with that.

Training notes (optional)
-------------------------

If you want to retrain the model from `IMDb Movies India.csv`, the typical steps used were:

- Read CSV with `encoding='latin1'` to avoid decoding errors.
- Clean `Year` column by stripping parentheses and converting to numeric.
- Impute or drop missing `Duration` values (median imputation works fine).
- Compute mean target-encoding for categorical columns: group by category and compute mean rating.
- Replace categories with their means and fill unseen categories at inference with `5.8`.
- Train `RandomForestRegressor` and save the model and encoders with `pickle`.

License & Attribution
---------------------

Include your preferred license or leave as-is. Dataset source: the CSV included in the folder.

Contact
-------

If you want any changes (move model to Releases, remove dataset, add `.gitignore`, or add a training script), tell me and I will update the repo and push the changes.

