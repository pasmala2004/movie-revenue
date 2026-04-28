import numpy as np
import xgboost as xgb
import pandas as pd

FEATURES = [
    'log_budget', 'runtime', 'popularity',
    'is_franchise', 'director_score',
    'release_month', 'release_dow',
    'is_summer', 'is_holiday',
    'genre_Action', 'genre_Comedy', 'genre_Drama',
    'genre_Thriller', 'genre_Animation', 'genre_Horror',
    'genre_Romance', 'genre_Adventure',
]

# Director score lookup — median log_revenue per director from training data
# Load once at startup
import os, pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgb_revenue_v1.json')

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)


def predict_revenue(
    budget: float,
    runtime: int,
    popularity: float,
    is_franchise: int,
    director_score: float,
    release_month: int,
    release_dow: int,
    genres: list[str],
) -> dict:

    is_summer  = 1 if release_month in [6, 7, 8]   else 0
    is_holiday = 1 if release_month in [11, 12]     else 0

    genre_flags = {
        'genre_Action':    1 if 'Action'    in genres else 0,
        'genre_Comedy':    1 if 'Comedy'    in genres else 0,
        'genre_Drama':     1 if 'Drama'     in genres else 0,
        'genre_Thriller':  1 if 'Thriller'  in genres else 0,
        'genre_Animation': 1 if 'Animation' in genres else 0,
        'genre_Horror':    1 if 'Horror'    in genres else 0,
        'genre_Romance':   1 if 'Romance'   in genres else 0,
        'genre_Adventure': 1 if 'Adventure' in genres else 0,
    }

    row = {
        'log_budget':     np.log1p(budget),
        'runtime':        runtime,
        'popularity':     popularity,
        'is_franchise':   is_franchise,
        'director_score': director_score,
        'release_month':  release_month,
        'release_dow':    release_dow,
        'is_summer':      is_summer,
        'is_holiday':     is_holiday,
        **genre_flags,
    }

    X = pd.DataFrame([row])[FEATURES]
    log_pred = model.predict(X)[0]
    revenue  = float(np.expm1(log_pred))

    # Confidence interval — model error factor is ~2.19x
    ERROR_FACTOR = 2.19
    return {
        'predicted_revenue': revenue,
        'low_estimate':      revenue / ERROR_FACTOR,
        'high_estimate':     revenue * ERROR_FACTOR,
        'log_prediction':    float(log_pred),
    }