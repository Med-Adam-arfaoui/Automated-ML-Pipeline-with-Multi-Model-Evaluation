from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from services.train_models import train_single_model
from services.preprocess_service import preprocess_data

router = APIRouter(prefix="/model", tags=["Model"])

class TrainRequest(BaseModel):
    data: list
    columns: list
    target: str
    model_name: str

@router.post("/train")
def train_model(req: TrainRequest):
    df = pd.DataFrame(req.data, columns=req.columns)
    cleaned_df, X, y = preprocess_data(df, req.target)
    result = train_single_model(req.model_name, X, y)
    return {"status": "success", "result": result}
