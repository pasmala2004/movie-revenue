# 🎬 Cinecast — Movie Revenue Predictor

> Predict opening weekend box office revenue before a single frame is shot.

**Live demo → [appapppy-cpieebtb6c8nwbeudzupen.streamlit.app](https://appapppy-cpieebtb6c8nwbeudzupen.streamlit.app/)**

Built on 3,213 films from the TMDB dataset. Full end-to-end ML pipeline: data engineering → EDA → XGBoost model (R²=0.80) → deployed Streamlit app with filmmaker intelligence dashboard.

---

## Live demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appapppy-cpieebtb6c8nwbeudzupen.streamlit.app/)

**Test it with Sinners (2025):**
- Budget: $90M · Runtime: 137 min · Popularity: 85
- Director score: 19.5 · Release: April · Genres: Horror, Thriller, Drama
- Model predicts: **$326.6M** · Actual worldwide: **$370M** ✓ (11% miss)

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
│  Streamlit Cloud    │  Live at streamlit.app
│  Filmmaker UI       │  Insights · What-if scenarios
└─────────────────────┘
```

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
│   └── streamlit_app.py            # Streamlit UI — Cinecast
├── models/
│   └── xgb_revenue_v1.json         # Trained XGBoost model
├── requirements.txt
└── README.md
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
jupyter nbconvert --to notebook --execute notebooks/stage1_data_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/stage2_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/stage3_model.ipynb
```

### 4. Run locally

```bash
streamlit run app/streamlit_app.py
```

---

## App features

**Revenue forecast** — low / predicted / high estimates with ±2.19× confidence interval.

**Model insights** — budget efficiency vs genre median, release window strength as % of peak month, franchise effect interpretation.

**Market context charts** — genre revenue comparison and seasonal calendar.

**Filmmaker recommendations** — actionable advice engine:
- Release timing warnings for weak months (Sep, Oct, Jan, Apr)
- Budget-to-ROI risk flags when return is below break-even
- Horror lean-budget guidance for maximum ROI
- Animation franchise and merchandise strategy
- Genre clarity guidance for marketing positioning
- Franchise leverage for original films

**What-if scenarios** — instant delta calculations: move to June, make it a franchise, cut budget 20%.

---

## Model performance

| Metric | Ridge baseline | XGBoost |
|---|---|---|
| R² | 0.7469 | **0.8020** |
| RMSE (log space) | 0.8860 | **0.7836** |
| MAE (log space) | 0.6357 | **0.5472** |
| Error factor | 2.43× | **2.19×** |
| CV Mean R² (5-fold) | — | **0.7584** |
| CV Std R² | — | **0.041** |

The model explains 80% of revenue variance. The irreducible ~20% is driven by signals outside the dataset — social media sentiment, pre-sale tickets, word of mouth, cultural timing.

**Real-world validation — Sinners (2025):**
- Inputs: $90M budget, April release, Horror/Thriller/Drama, standalone, Ryan Coogler
- Predicted: **$326.6M** · Actual: **$370M worldwide** · Miss: 11.7%

---

## Key findings from EDA

- **Budget is the strongest predictor** — correlation with revenue: 0.705
- **Horror has the best ROI** — 2.9× median return per dollar spent
- **Animation earns the most** in absolute terms — $197M median
- **Franchise films earn 1.95× standalone films** at the median
- **June is the best release month** — $112M median vs $27M in September
- **Revenue is log-normally distributed** — model predicts log(revenue)

---

## Stack

```
Python · Pandas · NumPy · PyArrow
XGBoost · scikit-learn · SHAP · MLflow
Streamlit · Matplotlib
```

---

## Stages status

| Stage | Status | Output |
|---|---|---|
| 1 · Data Engineering | ✅ Complete | `movies_clean.parquet` — 3,213 films |
| 2 · Data Analysis | ✅ Complete | `movies_analysis.parquet` + 8 EDA plots |
| 3 · ML Model | ✅ Complete | `xgb_revenue_v1.json` · R²=0.80 |
| 4 · Deploy | ✅ Complete | [Live on Streamlit Cloud](https://appapppy-cpieebtb6c8nwbeudzupen.streamlit.app/) |

---

## License

MIT