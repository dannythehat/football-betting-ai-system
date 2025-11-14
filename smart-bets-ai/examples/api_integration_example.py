"""
Example: Integrating Smart Bets AI with User API
This shows how to add Smart Bets predictions to the FastAPI application
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd

# Import Smart Bets AI modules
from smart_bets_ai.model_trainer import SmartBetsModelTrainer
from smart_bets_ai.predictor import SmartBetsPredictor, format_prediction_for_api

# Import existing modules
from data_ingestion.database import get_db
from data_ingestion.models import Match


# Initialize Smart Bets AI on application startup
class SmartBetsAIService:
    """Singleton service for Smart Bets AI"""
    
    def __init__(self):
        self.trainer = None
        self.predictor = None
        self.initialized = False
    
    def initialize(self, model_version: str = 'latest'):
        """Load models on startup"""
        if not self.initialized:
            print(f"Loading Smart Bets AI models (version: {model_version})...")
            self.trainer = SmartBetsModelTrainer()
            self.trainer.load_models(version=model_version)
            self.predictor = SmartBetsPredictor(self.trainer)
            self.initialized = True
            print("✓ Smart Bets AI models loaded successfully")
    
    def get_predictor(self) -> SmartBetsPredictor:
        """Get predictor instance"""
        if not self.initialized:
            raise RuntimeError("Smart Bets AI not initialized")
        return self.predictor


# Global service instance
smart_bets_service = SmartBetsAIService()


# FastAPI application
app = FastAPI(title="Football Betting AI System")


@app.on_event("startup")
async def startup_event():
    """Initialize Smart Bets AI on application startup"""
    smart_bets_service.initialize(model_version='v1.0')


# Helper function to convert Match ORM to DataFrame
def match_to_dataframe(match: Match) -> pd.DataFrame:
    """Convert SQLAlchemy Match object to DataFrame for prediction"""
    return pd.DataFrame([{
        'match_id': match.match_id,
        'home_team_id': match.home_team_id,
        'away_team_id': match.away_team_id,
        'match_datetime': match.match_datetime,
        'league': match.league,
        'season': match.season,
        'status': match.status,
        'home_goals_avg': float(match.home_goals_avg) if match.home_goals_avg else 0.0,
        'away_goals_avg': float(match.away_goals_avg) if match.away_goals_avg else 0.0,
        'home_goals_conceded_avg': float(match.home_goals_conceded_avg) if match.home_goals_conceded_avg else 0.0,
        'away_goals_conceded_avg': float(match.away_goals_conceded_avg) if match.away_goals_conceded_avg else 0.0,
        'home_corners_avg': float(match.home_corners_avg) if match.home_corners_avg else 0.0,
        'away_corners_avg': float(match.away_corners_avg) if match.away_corners_avg else 0.0,
        'home_cards_avg': float(match.home_cards_avg) if match.home_cards_avg else 0.0,
        'away_cards_avg': float(match.away_cards_avg) if match.away_cards_avg else 0.0,
        'home_btts_rate': float(match.home_btts_rate) if match.home_btts_rate else 0.0,
        'away_btts_rate': float(match.away_btts_rate) if match.away_btts_rate else 0.0,
        'home_form': match.home_form or '',
        'away_form': match.away_form or ''
    }])


# API Endpoints

@app.get("/api/v1/predictions/smart-bets/{match_id}")
async def get_smart_bet(
    match_id: str,
    include_all_markets: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get Smart Bet prediction for a specific match
    
    Args:
        match_id: Match identifier
        include_all_markets: Include all market probabilities in response
        db: Database session
        
    Returns:
        Smart Bet prediction with probability and alternatives
    """
    # Fetch match from database
    match = db.query(Match).filter(Match.match_id == match_id).first()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if match.status != 'scheduled':
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot predict completed match (status: {match.status})"
        )
    
    # Convert to DataFrame
    match_data = match_to_dataframe(match)
    
    # Generate prediction
    predictor = smart_bets_service.get_predictor()
    prediction = predictor.predict_match(match_data)
    
    # Format for API response
    formatted = format_prediction_for_api(prediction, include_all_markets)
    
    # Add match context
    formatted['match_info'] = {
        'match_id': match.match_id,
        'home_team_id': match.home_team_id,
        'away_team_id': match.away_team_id,
        'match_datetime': match.match_datetime.isoformat(),
        'league': match.league
    }
    
    return formatted


@app.get("/api/v1/predictions/smart-bets")
async def get_smart_bets_batch(
    limit: int = 10,
    league: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get Smart Bet predictions for upcoming matches
    
    Args:
        limit: Maximum number of matches to return
        league: Filter by league (optional)
        db: Database session
        
    Returns:
        List of Smart Bet predictions
    """
    # Query upcoming matches
    query = db.query(Match).filter(Match.status == 'scheduled')
    
    if league:
        query = query.filter(Match.league == league)
    
    matches = query.order_by(Match.match_datetime).limit(limit).all()
    
    if not matches:
        return {"predictions": [], "count": 0}
    
    # Convert to DataFrame
    matches_data = pd.concat([match_to_dataframe(m) for m in matches], ignore_index=True)
    
    # Generate predictions
    predictor = smart_bets_service.get_predictor()
    predictions = predictor.predict_batch(matches_data)
    
    # Format for API
    formatted_predictions = [
        format_prediction_for_api(pred, include_all_markets=False)
        for pred in predictions
    ]
    
    return {
        "predictions": formatted_predictions,
        "count": len(formatted_predictions)
    }


@app.post("/api/v1/predictions/analyze-custom-bet")
async def analyze_custom_bet(
    match_id: str,
    bet_type: str,
    db: Session = Depends(get_db)
):
    """
    Analyze a user-selected custom bet
    
    Args:
        match_id: Match identifier
        bet_type: Bet type to analyze (e.g., 'over_2_5', 'btts', 'home_win')
        db: Database session
        
    Returns:
        Custom bet analysis with verdict and comparison to Smart Bet
    """
    # Fetch match from database
    match = db.query(Match).filter(Match.match_id == match_id).first()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if match.status != 'scheduled':
        raise HTTPException(
            status_code=400,
            detail=f"Cannot analyze completed match (status: {match.status})"
        )
    
    # Convert to DataFrame
    match_data = match_to_dataframe(match)
    
    # Analyze custom bet
    predictor = smart_bets_service.get_predictor()
    
    try:
        analysis = predictor.analyze_custom_bet(match_data, bet_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Add match context
    analysis['match_info'] = {
        'match_id': match.match_id,
        'home_team_id': match.home_team_id,
        'away_team_id': match.away_team_id,
        'match_datetime': match.match_datetime.isoformat(),
        'league': match.league
    }
    
    return analysis


@app.get("/api/v1/predictions/model-info")
async def get_model_info():
    """
    Get information about loaded Smart Bets AI models
    
    Returns:
        Model metadata and performance metrics
    """
    if not smart_bets_service.initialized:
        raise HTTPException(status_code=503, detail="Smart Bets AI not initialized")
    
    summary = smart_bets_service.trainer.get_model_summary()
    
    return {
        "status": "loaded",
        "model_info": summary,
        "endpoints": {
            "smart_bet": "/api/v1/predictions/smart-bets/{match_id}",
            "batch_predictions": "/api/v1/predictions/smart-bets",
            "custom_analysis": "/api/v1/predictions/analyze-custom-bet"
        }
    }


# Example usage in main.py:
"""
# In user-api/main.py

from smart_bets_ai.model_trainer import SmartBetsModelTrainer
from smart_bets_ai.predictor import SmartBetsPredictor

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    global smart_bets_predictor
    
    # Load Smart Bets AI models
    trainer = SmartBetsModelTrainer()
    trainer.load_models(version='v1.0')
    smart_bets_predictor = SmartBetsPredictor(trainer)
    
    print("✓ Smart Bets AI loaded successfully")

# Use in endpoints
@app.get("/api/v1/predictions/smart-bets/{match_id}")
async def get_smart_bet(match_id: str, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.match_id == match_id).first()
    match_data = match_to_dataframe(match)
    prediction = smart_bets_predictor.predict_match(match_data)
    return prediction
"""
