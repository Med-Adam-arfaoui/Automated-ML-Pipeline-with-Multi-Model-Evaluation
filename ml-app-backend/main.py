from fastapi import FastAPI
from routes import data_routes, model_routes, analysis_routes

app = FastAPI(title="ML Project Backend", version="1.0")

# Register routers
app.include_router(data_routes.router)
app.include_router(model_routes.router)
app.include_router(analysis_routes.router)

@app.get("/")
def root():
    return {"message": "Welcome to the ML Project API"}

# Run using: uvicorn main:app --reload
