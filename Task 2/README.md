# IMDb Movie Rating Predictor

Simple Streamlit app that predicts IMDb movie ratings using a trained Random Forest model.

Files in this repository:

- `app.py` — Streamlit application (single-file, beginner style)
- `movie_model.pkl` — trained model (binary pickle)
- `target_encoders.pkl` — dictionary with target encodings for categorical features
- `requirements.txt` — Python dependencies
- `IMDb Movies India.csv` — original dataset (optional)

Run locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Notes:
- If `movie_model.pkl` or `target_encoders.pkl` are missing, the app will show an error.
- For GitHub: you may want to add `movie_model.pkl` and large datasets to `.gitignore` if you prefer not to store large binaries in the repo.

Author: Internship project
