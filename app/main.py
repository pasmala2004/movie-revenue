from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_revenue

app = FastAPI(title="Movie Revenue Predictor", version="1.0")


class MovieInput(BaseModel):
    title:          str
    budget:         float
    runtime:        int
    popularity:     float = 10.0
    is_franchise:   int   = 0
    director_score: float = 17.5   # median director score
    release_month:  int   = 6
    release_dow:    int   = 4      # Friday
    genres:         list[str] = []


class PredictionOutput(BaseModel):
    title:             str
    predicted_revenue: float
    low_estimate:      float
    high_estimate:     float
    predicted_revenue_fmt: str
    low_estimate_fmt:      str
    high_estimate_fmt:     str


def fmt(n):
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    return f"${n/1e6:.1f}M"


@app.get("/")
def root():
    return {"status": "ok", "model": "xgb_revenue_v1"}


@app.post("/predict", response_model=PredictionOutput)
def predict(movie: MovieInput):
    result = predict_revenue(
        budget         = movie.budget,
        runtime        = movie.runtime,
        popularity     = movie.popularity,
        is_franchise   = movie.is_franchise,
        director_score = movie.director_score,
        release_month  = movie.release_month,
        release_dow    = movie.release_dow,
        genres         = movie.genres,
    )
    return PredictionOutput(
        title              = movie.title,
        predicted_revenue  = result['predicted_revenue'],
        low_estimate       = result['low_estimate'],
        high_estimate      = result['high_estimate'],
        predicted_revenue_fmt = fmt(result['predicted_revenue']),
        low_estimate_fmt      = fmt(result['low_estimate']),
        high_estimate_fmt     = fmt(result['high_estimate']),
    )