from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
from services.ai_analysis_service import generate_analysis

router = APIRouter(prefix="/analysis", tags=["AI Analysis"])

class AnalysisRequest(BaseModel):
    raw_data: List[List[Any]]
    raw_columns: List[str]
    preprocessed_data: List[List[Any]] 
    preprocessed_columns: List[str]
    stored_models: Dict[str, Any]
    target_col: str = None

@router.post("/anal")
async def ai_analysis(request: AnalysisRequest):
    try:
        # Convert data back to DataFrames
        raw_df = pd.DataFrame(request.raw_data, columns=request.raw_columns)
        preprocessed_df = pd.DataFrame(request.preprocessed_data, columns=request.preprocessed_columns)
        
        # Generate analysis
        analysis = generate_analysis(
            raw_df=raw_df,
            preprocessed_df=preprocessed_df,
            stored_models=request.stored_models,
            target_col=request.target_col
        )
        
        return {"analysis": analysis}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "AI Analysis service is healthy"}