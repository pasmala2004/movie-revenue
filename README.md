# Movie Revenue Prediction Pipeline

An end-to-end machine learning system that predicts a film's opening weekend box office revenue from pre-release metadata — budget, cast, genre, director, and release timing. Built on the TMDB 5000 Movie Dataset.

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
│   ├── stage3_model.ipynb          # coming in Stage 3
│   └── stage4_deploy.ipynb         # coming in Stage 4
├── src/                            # Reusable modules (coming in Stage 3+)
├── models/                         # Saved model artifacts (coming in Stage 3)
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
│  Clean · Transform  │  4,803 → 3,213 films
│  Save to Parquet    │  23 features
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
│  Stage 3            │  ML Model  ← in progress
│  Train · Tune       │  XGBoost regressor
│  Evaluate · Explain │  SHAP explainability
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 4            │  Deployment  ← coming soon
│  FastAPI endpoint   │  Streamlit demo UI
│  Docker · Monitor   │  Drift monitoring
└─────────────────────┘
         │
         ▼
  Predicted opening weekend revenue ($)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/pasmala2004/movie-revenue.git
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
```

Or open the notebooks interactively in Jupyter and run cell by cell.

---

## Requirements

```
pandas
numpy
pyarrow
matplotlib
scikit-learn
xgboost
shap
mlflow
fastapi
uvicorn
streamlit
jupyter
```

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## Dataset

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) via Kaggle

**Files used:**
- `tmdb_5000_movies.csv` — budget, revenue, genres, release date, keywords, runtime, popularity
- `tmdb_5000_credits.csv` — full cast and crew with TMDB popularity scores

**License:** Data sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/) API under their terms of use.

---

## Features

| Feature | Source | Description |
|---|---|---|
| `budget` | TMDB | Production budget in USD |
| `log_budget` | Engineered | log(budget) — used as model input |
| `runtime` | TMDB | Film length in minutes |
| `popularity` | TMDB | Pre-release TMDB trending score |
| `cast_popularity` | Engineered | Sum of top 3 billed actors' popularity scores |
| `director_score` | Engineered | Director's median log revenue across all their films |
| `is_franchise` | Engineered | 1 if film belongs to a franchise or sequel, 0 otherwise |
| `release_month` | Engineered | Month of theatrical release (1–12) |
| `release_dow` | Engineered | Day of week of release (0=Mon, 4=Fri) |
| `is_summer` | Engineered | 1 if released June–August |
| `is_holiday` | Engineered | 1 if released November–December |
| `genre_*` | Engineered | Binary flags for 8 top genres |

**Target variable:** `log_revenue` — log-transformed box office revenue. Predictions are exponentiated back to USD at inference time.

---

## Key findings from EDA (Stage 2)

- **Budget is the strongest predictor** — log-log correlation with revenue is ~0.75
- **Horror has the best ROI** — lowest budgets, consistent returns
- **Animation and Adventure earn the most** in absolute revenue terms
- **Franchise films earn roughly 2–3× standalone films** at the median
- **Summer releases earn roughly 1.5–2× off-peak releases**
- **Revenue is log-normally distributed** — model predicts log(revenue), not raw revenue

---

## Stages status

| Stage | Status | Output |
|---|---|---|
| 1 · Data Engineering | ✅ Complete | `movies_clean.parquet` |
| 2 · Data Analysis | ✅ Complete | `movies_analysis.parquet` + 6 plots |
| 3 · ML Model | 🔄 In progress | — |
| 4 · Deploy | ⏳ Pending | — |

---

## License

MIT