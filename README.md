# Movie Revenue Prediction Pipeline

An end-to-end machine learning system that predicts a film's opening weekend box office revenue from pre-release metadata — budget, genre, director, and release timing. Built on the TMDB 5000 Movie Dataset.

---

## Project structure

```
movie-revenue/
├── data/
│   ├── raw/                        # Original CSVs from Kaggle (not committed)
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/                  # Cleaned and engineered datasets
│       ├── movies_clean.parquet    # Output of Stage 1
│       └── movies_analysis.parquet # Output of Stage 2
├── notebooks/
│   ├── stage1_data_engineering.ipynb
│   ├── stage2_analysis.ipynb
│   └── stage3_model.ipynb
├── app/
│   ├── main.py                     # FastAPI backend
│   ├── predict.py                  # Prediction logic
│   └── streamlit_app.py            # Frontend UI — Cinecast
├── models/
│   └── xgb_revenue_v1.json         # Trained XGBoost model
├── requirements.txt
└── README.md
```

---

## Pipeline overview

```
TMDB 5000 Dataset (Kaggle)
        │
        ▼
┌─────────────────────┐
│  Stage 1            │  Data Engineering
│  Clean · Transform  │  4,803 → 3,213 films · 23 features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2            │  Data Analysis & EDA
│  Explore · Visualise│  Revenue patterns, genre ROI,
│  Feature engineer   │  seasonal effects, director scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3            │  ML Model
│  Train · Tune       │  XGBoost · R²=0.80
│  Evaluate · Explain │  SHAP explainability
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 4            │  Deployment ✅
│  FastAPI endpoint   │  REST API · Streamlit UI
│  Cinecast UI        │  Filmmaker insights · What-if scenarios
└─────────────────────┘
         │
         ▼
  Predicted opening weekend revenue ($)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/movie-revenue.git
cd movie-revenue
pip install -r requirements.txt
```

### 2. Download the data

Go to [kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and download:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place both files in `data/raw/`.

### 3. Run the pipeline

```bash
# Stage 1 — data engineering
jupyter nbconvert --to notebook --execute notebooks/stage1_data_engineering.ipynb

# Stage 2 — analysis
jupyter nbconvert --to notebook --execute notebooks/stage2_analysis.ipynb

# Stage 3 — train model
jupyter nbconvert --to notebook --execute notebooks/stage3_model.ipynb
```

### 4. Launch the app

Open two terminals from the project root:

```bash
# Terminal 1 — API
cd app && uvicorn main:app --reload --port 8000

# Terminal 2 — UI
cd app && streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## App features (Cinecast UI)

**Revenue forecast** — low / predicted / high estimates with a ±2.19× confidence interval.

**Model insights** — three contextual cards explaining each prediction: budget efficiency vs genre median, release window strength as % of peak month, and franchise effect.

**Market context charts** — genre revenue comparison and seasonal calendar showing where the film sits historically.

**Filmmaker recommendations** — actionable advice engine covering:
- Release timing warnings for weak months (Sep, Oct, Jan, Apr)
- Budget-to-ROI risk flags when return is below break-even
- Horror-specific guidance on staying lean for maximum ROI
- Animation franchise and merchandise strategy
- Genre clarity guidance for marketing positioning
- Franchise leverage opportunities for original films

**What-if scenarios** — instant delta calculations: move to June, make it a franchise entry, cut budget 20%.

---

## Requirements

```
pandas · numpy · pyarrow · matplotlib
scikit-learn · xgboost · shap · mlflow
fastapi · uvicorn · streamlit · pydantic · requests · jupyter
```

```bash
pip install -r requirements.txt
```

---

## Dataset

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) via Kaggle

**License:** Data sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/) under their terms of use.

---

## Features used by the model

| Feature | Source | Description |
|---|---|---|
| `log_budget` | Engineered | log(budget) — top feature by SHAP |
| `runtime` | TMDB | Film length in minutes |
| `popularity` | TMDB | Pre-release TMDB trending score |
| `director_score` | Engineered | Director's median log revenue across their films |
| `is_franchise` | Engineered | 1 if sequel or franchise entry |
| `release_month` | Engineered | Month of theatrical release (1–12) |
| `release_dow` | Engineered | Day of week (0=Mon, 4=Fri) |
| `is_summer` | Engineered | 1 if released June–August |
| `is_holiday` | Engineered | 1 if released November–December |
| `genre_*` | Engineered | Binary flags for 8 genres |

**Target:** `log_revenue` — exponentiated back to USD at inference time.

---

## Key findings from EDA (Stage 2)

- **Budget is the strongest predictor** — correlation with revenue: 0.705
- **Horror has the best ROI** — 2.9× median return per dollar spent
- **Animation earns the most** in absolute terms — $197M median
- **Franchise films earn 1.95× standalone films** at the median
- **June is the best release month** — $112M median vs $27M in September
- **Revenue is log-normally distributed** — model predicts log(revenue)

---

## Model performance (Stage 3)

| Metric | Ridge baseline | XGBoost |
|---|---|---|
| R² | 0.7469 | **0.8020** |
| RMSE (log space) | 0.8860 | **0.7836** |
| MAE (log space) | 0.6357 | **0.5472** |
| Error factor | 2.43× | **2.19×** |
| CV Mean R² (5-fold) | — | **0.7584** |
| CV Std R² | — | **0.041** |

The model explains 80% of revenue variance and is stable across all cross-validation folds. The irreducible ~20% is driven by signals outside the dataset — social media, pre-sale tickets, word of mouth, cultural timing.

---

## Stages status

| Stage | Status | Output |
|---|---|---|
| 1 · Data Engineering | ✅ Complete | `movies_clean.parquet` — 3,213 films |
| 2 · Data Analysis | ✅ Complete | `movies_analysis.parquet` + 8 EDA plots |
| 3 · ML Model | ✅ Complete | `xgb_revenue_v1.json` · R²=0.80 |
| 4 · Deploy | ✅ Complete | FastAPI + Cinecast Streamlit UI |

---

## License

MIT