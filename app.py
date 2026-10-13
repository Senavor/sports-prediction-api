import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SportsPredictor")

# --- Model & Data Structures ---
MODEL_PATH = os.getenv("MODEL_PATH", "final_unified_sports_predictor.joblib")


class UnifiedSportsPredictor:
    """Reconstructed class definition to enable joblib deserialization."""
    def __init__(
        self, 
        model_1x2: Any, 
        le_1x2: Any, 
        model_btts: Any, 
        model_ou15: Any, 
        model_ou25: Any, 
        reference_data: pd.DataFrame
    ):
        self.model_1x2 = model_1x2
        self.le_1x2 = le_1x2
        self.model_btts = model_btts
        self.model_ou15 = model_ou15
        self.model_ou25 = model_ou25
        self.reference_data = reference_data


# FIX: Patch sys.modules so joblib can resolve __main__.UnifiedSportsPredictor
sys.modules["__main__"].UnifiedSportsPredictor = UnifiedSportsPredictor


# Global container for state
state: Dict[str, Optional[UnifiedSportsPredictor]] = {"predictor": None}


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory on startup and clean up resources on shutdown."""
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading unified sports predictor from '{MODEL_PATH}'...")
            state["predictor"] = joblib.load(MODEL_PATH)
            logger.info("🚀 Success: Model loaded and ready for predictions!")
        except Exception as e:
            logger.critical(f"❌ Failed to load model file: {e}", exc_info=True)
    else:
        logger.warning(
            f"⚠️ Model file '{MODEL_PATH}' not found. "
            "Server will start, but /predict endpoint will return 500 errors."
        )
    yield
    # Cleanup logic during shutdown
    state["predictor"] = None
    logger.info("Server shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Unified Sports Prediction API",
    description="Optimized FastAPI engine for match outcomes, BTTS, and goal-line predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production (e.g., ["https://yourdomain.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Validation Schema ---
class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"], description="Home team name")
    away_team: str = Field(..., examples=["Chelsea"], description="Away team name")
    b365_home_odds: float = Field(..., gt=1.0, examples=[1.95], description="Bet365 Home odds")
    b365_draw_odds: float = Field(..., gt=1.0, examples=[3.40], description="Bet365 Draw odds")
    b365_away_odds: float = Field(..., gt=1.0, examples=[3.80], description="Bet365 Away odds")


# --- Helper Functions ---
def extract_metrics(target_team: str, row_df: pd.DataFrame):
    """Extract rolling performance metrics dynamically based on home/away status."""
    row = row_df.iloc[0]
    if row["HomeTeam"].lower() == target_team.lower():
        return (
            row["Home_Rolling_Goals_Scored"],
            row["Home_Rolling_Goals_Conceded"],
            row["Home_Rolling_Shots"],
            row["Home_Points_Momentum"],
        )
    return (
        row["Away_Rolling_Goals_Scored"],
        row["Away_Rolling_Goals_Conceded"],
        row["Away_Rolling_Shots"],
        row["Away_Points_Momentum"],
    )


# --- API Routes ---
@app.get("/", tags=["Health Check"])
def index():
    """Health check endpoint to verify server status and model loading."""
    predictor_loaded = state["predictor"] is not None
    return {
        "status": "online",
        "model_loaded": predictor_loaded,
        "message": (
            "Sports Predictor Engine is running!"
            if predictor_loaded
            else "Server online, but model is not loaded."
        ),
    }


@app.post("/predict", tags=["Predictions"])
def predict(payload: MatchPredictionRequest):
    """Generate match outcome probabilities and goal lines."""
    predictor = state["predictor"]

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model is not loaded on the server. Verify '{MODEL_PATH}' exists.",
        )

    home = payload.home_team.strip()
    away = payload.away_team.strip()

    ref_data = predictor.reference_data

    # Query historical reference records
    latest_home = ref_data[
        (ref_data["HomeTeam"].str.lower() == home.lower())
        | (ref_data["AwayTeam"].str.lower() == home.lower())
    ].tail(1)

    latest_away = ref_data[
        (ref_data["HomeTeam"].str.lower() == away.lower())
        | (ref_data["AwayTeam"].str.lower() == away.lower())
    ].tail(1)

    if latest_home.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Form data not found for Home Team: '{home}'. Check team spelling.",
        )
    if latest_away.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Form data not found for Away Team: '{away}'. Check team spelling.",
        )

    # Extract team metrics
    h_g, h_gc, h_s, h_m = extract_metrics(home, latest_home)
    a_g, a_gc, a_s, a_m = extract_metrics(away, latest_away)

    # Calculate Head-to-Head (H2H) stats
    h2h_matches = ref_data[
        (
            (ref_data["HomeTeam"].str.lower() == home.lower())
            & (ref_data["AwayTeam"].str.lower() == away.lower())
        )
        | (
            (ref_data["HomeTeam"].str.lower() == away.lower())
            & (ref_data["AwayTeam"].str.lower() == home.lower())
        )
    ]

    h2h_win_diff = 1.0
    h2h_g_diff = 0.0

    if not h2h_matches.empty:
        goal_diffs = []
        win_pts = []
        for _, match in h2h_matches.iterrows():
            if match["HomeTeam"].lower() == home.lower():
                goal_diffs.append(match["FTHG"] - match["FTAG"])
                win_pts.append(3 if match["FTR"] == "H" else (1 if match["FTR"] == "D" else 0))
            else:
                goal_diffs.append(match["FTAG"] - match["FTHG"])
                win_pts.append(3 if match["FTR"] == "A" else (1 if match["FTR"] == "D" else 0))
        h2h_win_diff = float(np.mean(win_pts))
        h2h_g_diff = float(np.mean(goal_diffs))

    # Assemble feature vector matching LightGBM / model layout
    input_features = pd.DataFrame([
        {
            "Home_Rolling_Goals_Scored": h_g,
            "Away_Rolling_Goals_Scored": a_g,
            "Home_Rolling_Goals_Conceded": h_gc,
            "Away_Rolling_Goals_Conceded": a_gc,
            "Home_Rolling_Shots": h_s,
            "Away_Rolling_Shots": a_s,
            "Home_Points_Momentum": h_m,
            "Away_Points_Momentum": a_m,
            "H2H_Win_Rate_Diff": h2h_win_diff,
            "H2H_Avg_Goal_Diff": h2h_g_diff,
            "Diff_Goals_Scored": h_g - a_g,
            "Diff_Goals_Conceded": h_gc - a_gc,
            "Diff_Rolling_Shots": h_s - a_s,
            "Diff_Points_Momentum": h_m - a_m,
            "B365H": payload.b365_home_odds,
            "B365D": payload.b365_draw_odds,
            "B365A": payload.b365_away_odds,
        }
    ])

    # Model inference
    probs_1x2 = predictor.model_1x2.predict_proba(input_features)[0]
    prob_btts = predictor.model_btts.predict_proba(input_features)[0][1]
    prob_ou15 = predictor.model_ou15.predict_proba(input_features)[0][1]
    prob_ou25 = predictor.model_ou25.predict_proba(input_features)[0][1]

    prob_h, prob_d, prob_a = (
        float(probs_1x2[2]),
        float(probs_1x2[1]),
        float(probs_1x2[0]),
    )

    return {
        "success": True,
        "matchup": f"{home} vs {away}",
        "h2h_matches_analyzed": len(h2h_matches),
        "probabilities": {
            "outcome_1x2": {
                "home_win": round(prob_h, 4),
                "draw": round(prob_d, 4),
                "away_win": round(prob_a, 4),
            },
            "double_chance": {
                "dc_1X": round(prob_h + prob_d, 4),
                "dc_12": round(prob_h + prob_a, 4),
                "dc_X2": round(prob_d + prob_a, 4),
            },
            "goal_lines": {
                "btts_yes": round(prob_btts, 4),
                "btts_no": round(1 - prob_btts, 4),
                "over_1_5": round(prob_ou15, 4),
                "under_1_5": round(1 - prob_ou15, 4),
                "over_2_5": round(prob_ou25, 4),
                "under_2_5": round(1 - prob_ou25, 4),
            },
        },
    }