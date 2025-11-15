"""
Football Betting AI System - User API
FastAPI application serving predictions and handling data ingestion
"""

import os
from typing import List, Dict, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.database import get_db_session, init_db
from data_ingestion.schemas import BatchIngestRequest, IngestResponse
from data_ingestion.ingestion import DataIngestionService

# Import Smart Bets predictor
try:
    from smart_bets_ai.predict import SmartBetsPredictor
    SMART_BETS_AVAILABLE = True
except ImportError:
    SMART_BETS_AVAILABLE = False
    print("⚠️  Smart Bets AI not available. Train models first.")

# Import Golden Bets predictor
try:
    from golden_bets_ai.predict import GoldenBetsPredictor
    GOLDEN_BETS_AVAILABLE = True
except ImportError:
    GOLDEN_BETS_AVAILABLE = False
    print("⚠️  Golden Bets AI not available. Train models first.")

# Import Value Bets predictor
try:
    from value_bets_ai.predict import ValueBetsPredictor
    VALUE_BETS_AVAILABLE = True
except ImportError:
    VALUE_BETS_AVAILABLE = False
    print("⚠️  Value Bets AI not available.")

# Import Custom Analysis
try:
    from custom_analysis import CustomBetAnalyzer
    CUSTOM_ANALYSIS_AVAILABLE = True
except ImportError:
    CUSTOM_ANALYSIS_AVAILABLE = False
    print("⚠️  Custom Analysis not available.")

# Initialize FastAPI app
app = FastAPI(
    title="Football Betting AI System",
    description="AI prediction engine for football betting intelligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictors
predictor = None
golden_predictor = None
value_predictor = None
custom_analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and models on startup"""
    global predictor, golden_predictor, value_predictor, custom_analyzer
    
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    # Load Smart Bets models
    if SMART_BETS_AVAILABLE:
        try:
            predictor = SmartBetsPredictor()
            print("✅ Smart Bets AI models loaded")
        except Exception as e:
            print(f"⚠️  Could not load Smart Bets models: {e}")
    
    # Load Golden Bets models
    if GOLDEN_BETS_AVAILABLE:
        try:
            golden_predictor = GoldenBetsPredictor()
            print("✅ Golden Bets AI models loaded")
        except Exception as e:
            print(f"⚠️  Could not load Golden Bets models: {e}")
    
    # Load Value Bets models
    if VALUE_BETS_AVAILABLE:
        try:
            value_predictor = ValueBetsPredictor()
            print("✅ Value Bets AI models loaded")
        except Exception as e:
            print(f"⚠️  Could not load Value Bets models: {e}")
    
    # Load Custom Analysis
    if CUSTOM_ANALYSIS_AVAILABLE:
        try:
            custom_analyzer = CustomBetAnalyzer()
            print("✅ Custom Analysis loaded")
        except Exception as e:
            print(f"⚠️  Could not load Custom Analysis: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Football Betting AI System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "data_ingestion": "/api/v1/data/ingest",
            "smart_bets": "/api/v1/predictions/smart-bets",
            "golden_bets": "/api/v1/predictions/golden-bets",
            "value_bets": "/api/v1/predictions/value-bets",
            "custom_analysis": "/api/v1/predictions/custom-analysis",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "football-betting-ai",
        "version": "1.0.0",
        "smart_bets_available": predictor is not None,
        "golden_bets_available": golden_predictor is not None,
        "value_bets_available": value_predictor is not None,
        "custom_analysis_available": custom_analyzer is not None
    }


@app.post(
    "/api/v1/data/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["Data Ingestion"]
)
async def ingest_data(
    request: BatchIngestRequest,
    db: Session = Depends(get_db_session)
):
    """
    Ingest match data from main application
    
    Accepts:
    - Historical matches with results
    - Upcoming fixtures without results
    - Team statistics
    - Betting odds
    
    Returns:
    - Processing statistics
    - Any errors encountered
    """
    try:
        service = DataIngestionService(db)
        response = service.ingest_batch(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": response.message,
                    "errors": response.errors
                }
            )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Pydantic models for predictions
class MatchInput(BaseModel):
    """Match data for prediction"""
    match_id: str
    home_team: str
    away_team: str
    home_goals_avg: float
    away_goals_avg: float
    home_goals_conceded_avg: float
    away_goals_conceded_avg: float
    home_corners_avg: float
    away_corners_avg: float
    home_cards_avg: float
    away_cards_avg: float
    home_btts_rate: float
    away_btts_rate: float
    home_form: str = ""
    away_form: str = ""


class MatchWithOdds(MatchInput):
    """Match data with odds for value betting"""
    odds: Dict[str, float]


class PredictionRequest(BaseModel):
    """Request for predictions"""
    matches: List[MatchInput]


class ValueBetsRequest(BaseModel):
    """Request for Value Bets predictions"""
    matches: List[MatchWithOdds]


class CustomAnalysisRequest(BaseModel):
    """Request for Custom Bet Analysis"""
    match_data: MatchInput
    market_id: str
    selection_id: str


@app.post(
    "/api/v1/predictions/smart-bets",
    tags=["Predictions"],
    status_code=status.HTTP_200_OK
)
async def get_smart_bets(request: PredictionRequest):
    """
    Get Smart Bets predictions for matches
    
    Returns the highest probability bet across all 4 markets for each fixture:
    - Goals: Over/Under 2.5
    - Cards: Over/Under 3.5
    - Corners: Over/Under 9.5
    - BTTS: Yes/No
    
    Each prediction includes:
    - Market and selection details
    - Probability and percentage
    - Explanation
    - Alternative markets with probabilities
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Smart Bets AI models not loaded. Please train models first."
        )
    
    try:
        # Convert Pydantic models to dicts
        matches = [match.model_dump() for match in request.matches]
        
        # Get predictions
        predictions = predictor.predict_batch(matches)
        
        return {
            "success": True,
            "total_matches": len(matches),
            "predictions": predictions,
            "model_version": predictor.metadata.get('version', '1.0.0')
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post(
    "/api/v1/predictions/golden-bets",
    tags=["Predictions"],
    status_code=status.HTTP_200_OK
)
async def predict_golden_bets(request: PredictionRequest):
    """
    Generate Golden Bets predictions (1-3 daily picks, 85%+ confidence)
    
    Golden Bets are the highest confidence predictions filtered from Smart Bets:
    - Maximum 3 predictions per day
    - Minimum 85% confidence score
    - High ensemble model agreement
    - Detailed reasoning for each pick
    
    Returns:
    - Top 1-3 Golden Bets with highest confidence
    - Confidence scores and ensemble agreement
    - Detailed reasoning for each prediction
    """
    if golden_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Golden Bets AI models not loaded. Please train models first."
        )
    
    try:
        # Convert Pydantic models to dicts
        matches = [match.model_dump() for match in request.matches]
        
        # Get Golden Bets predictions
        predictions = golden_predictor.predict(matches)
        
        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "max_daily": 3
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Golden Bets prediction error: {str(e)}"
        )


@app.post(
    "/api/v1/predictions/value-bets",
    tags=["Predictions"],
    status_code=status.HTTP_200_OK
)
async def predict_value_bets(request: ValueBetsRequest):
    """
    Generate Value Bets predictions (top 3 daily picks with positive EV)
    
    Value Bets identify opportunities where AI probability exceeds bookmaker odds:
    - Maximum 3 predictions per day
    - Minimum 10% positive value
    - Minimum 5% expected value (EV)
    - Focus on long-term profitability
    
    Required: Match data with odds for all 8 selections:
    - goals_over_2_5, goals_under_2_5
    - cards_over_3_5, cards_under_3_5
    - corners_over_9_5, corners_under_9_5
    - btts_yes, btts_no
    
    Returns:
    - Top 3 Value Bets ranked by value score
    - AI probability vs implied probability
    - Value percentage and expected value
    - Detailed reasoning for each pick
    """
    if value_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Value Bets AI not loaded."
        )
    
    try:
        # Convert Pydantic models to dicts
        matches = [match.model_dump() for match in request.matches]
        
        # Get Value Bets predictions
        predictions = value_predictor.predict(matches)
        
        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "max_daily": 3
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Value Bets prediction error: {str(e)}"
        )


@app.post(
    "/api/v1/predictions/custom-analysis",
    tags=["Predictions"],
    status_code=status.HTTP_200_OK
)
async def analyze_custom_bet(request: CustomAnalysisRequest):
    """
    Analyze a user-selected bet (Custom Bet Analysis)
    
    Provides on-demand analysis for user-selected fixtures and bet types:
    - User chooses any fixture and one of the 4 markets
    - AI analyzes the specific bet
    - Returns probability, verdict, and reasoning
    - Compares with Smart Bet recommendation
    - Educational feedback on bet quality
    
    Supported markets:
    - total_goals: over_2.5, under_2.5
    - total_cards: over_3.5, under_3.5
    - total_corners: over_9.5, under_9.5
    - btts: yes, no
    
    Returns:
    - Probability and confidence level
    - Verdict (Excellent/Good/Moderate/Weak/Not Recommended)
    - Market-specific explanation
    - Comparison with Smart Bet
    - Alternative Smart Bet suggestion if different
    """
    if custom_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Custom Analysis not loaded. Please ensure Smart Bets models are trained."
        )
    
    try:
        # Convert match data to dict
        match_data = request.match_data.model_dump()
        
        # Analyze custom bet
        result = custom_analyzer.analyze_custom_bet(
            match_data=match_data,
            market_id=request.market_id,
            selection_id=request.selection_id
        )
        
        return {
            "success": True,
            "analysis": result
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom Analysis error: {str(e)}"
        )


@app.get("/api/v1/matches", tags=["Matches"])
async def get_matches(
    limit: int = 10,
    status: str = "scheduled",
    db: Session = Depends(get_db_session)
):
    """
    Get matches from database
    
    Query parameters:
    - limit: Number of matches to return (default: 10)
    - status: Match status filter (scheduled, completed, all)
    """
    from data_ingestion.models import Match, Team
    
    query = db.query(Match).join(
        Team, Match.home_team_id == Team.team_id
    )
    
    if status != "all":
        query = query.filter(Match.status == status)
    
    matches = query.order_by(Match.match_datetime).limit(limit).all()
    
    return {
        "total": len(matches),
        "matches": [
            {
                "match_id": m.match_id,
                "home_team": m.home_team.team_name,
                "away_team": m.away_team.team_name,
                "match_datetime": m.match_datetime.isoformat(),
                "league": m.league,
                "status": m.status
            }
            for m in matches
        ]
    }


@app.get("/api/v1/teams", tags=["Teams"])
async def get_teams(
    limit: int = 20,
    db: Session = Depends(get_db_session)
):
    """Get teams from database"""
    from data_ingestion.models import Team
    
    teams = db.query(Team).limit(limit).all()
    
    return {
        "total": len(teams),
        "teams": [
            {
                "team_id": t.team_id,
                "team_name": t.team_name,
                "league": t.league,
                "tier": t.tier
            }
            for t in teams
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )
