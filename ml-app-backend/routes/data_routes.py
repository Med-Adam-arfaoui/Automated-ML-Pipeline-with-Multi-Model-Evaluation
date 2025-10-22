from fastapi import APIRouter, UploadFile, File, Form
import pandas as pd
import io
from fastapi.responses import JSONResponse, StreamingResponse
from services.preprocess_service import preprocess_data 

router = APIRouter(prefix="/data", tags=["Data"])

# Store uploaded CSV in memory
uploaded_data = {}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        uploaded_data["df"] = df  # store DataFrame in memory
        return {"message": "File uploaded successfully", "columns": df.columns.tolist()}
    except Exception as e:
        return {"error": str(e)}
    

@router.post("/preprocess")
async def preprocess_endpoint(target_col: str = Form(...)):
    if "df" not in uploaded_data:
        return JSONResponse(content={"error": "No data uploaded yet"}, status_code=400)
    
    try:
        df = uploaded_data["df"]

        # unpack tuple (df_cleaned, X, y)
        df_cleaned, X, y = preprocess_data(df, target_col)

        # store for later use
        uploaded_data["cleaned_df"] = df_cleaned
        uploaded_data["X"] = X
        uploaded_data["y"] = y

        # only send the cleaned full DataFrame back as CSV
        stream = io.StringIO()
        df_cleaned.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(stream, media_type="text/csv")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
