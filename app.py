"""
FastAPI Web Service & REST API for News Article Classification.

Provides:
- Web UI frontend served at '/'
- Single and batch prediction endpoints ('/predict', '/api/predict')
- Interactive Swagger documentation at '/docs'
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.predict import NewsClassifier
from src.config import CLASS_NAMES
from src import __version__

# Base directory
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

# Initialize FastAPI application
app = FastAPI(
    title="News Classification System",
    description=(
        "Production-ready NLP API and Web Interface that classifies news articles into "
        "4 primary categories: World, Sports, Business, and Sci/Tech."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the singleton classifier on startup
classifier = NewsClassifier()


from src.schemas import (
    SingleArticleRequest,
    BatchArticleRequest,
    SingleArticleResponse,
    HealthResponse,
)


# =====================================================================
# Core Prediction Logic
# =====================================================================

def execute_prediction(title: Optional[str] = None, description: Optional[str] = "", text: Optional[str] = None) -> Dict:
    if not classifier.pipeline:
        detail_msg = classifier.load_error or "Model is not loaded. Ensure models/news_classifier_pipeline.pkl exists."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail_msg
        )

    # Support single text block input
    if text and not title:
        parts = text.strip().split("\n", 1)
        title = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
    elif not title and not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide news article text or a title."
        )

    try:
        return classifier.predict(title=title or "", description=description or "")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# =====================================================================
# Endpoints (Supporting both root and /api/ prefixes for Vercel)
# =====================================================================

@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["System"])
@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def health_check():
    """
    Validates model pipeline readiness and API health.
    """
    is_loaded = classifier.pipeline is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "version": __version__,
        "model_loaded": is_loaded,
        "categories": CLASS_NAMES,
        "error": classifier.load_error if not is_loaded else None,
    }


@app.post(
    "/predict",
    response_model=SingleArticleResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a Single News Article",
    tags=["Inference"]
)
@app.post("/api/predict", response_model=SingleArticleResponse, include_in_schema=False)
def predict_single(request: SingleArticleRequest):
    """
    Classifies a news article into one of 4 categories and returns confidence probabilities.
    """
    return execute_prediction(request.title, request.description, request.text)


@app.post(
    "/predict/batch",
    response_model=List[SingleArticleResponse],
    status_code=status.HTTP_200_OK,
    summary="Classify a Batch of News Articles",
    tags=["Inference"]
)
@app.post("/api/predict/batch", response_model=List[SingleArticleResponse], include_in_schema=False)
def predict_batch(request: BatchArticleRequest):
    """
    Batch classification endpoint for processing multiple news articles simultaneously.
    """
    if not classifier.pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    return [execute_prediction(item.title, item.description) for item in request.articles]


# =====================================================================
# Serve Frontend Static UI
# =====================================================================

if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

    @app.get("/", summary="Web Application UI", tags=["Frontend"])
    def serve_frontend():
        index_file = PUBLIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "News Classification API is running. Visit /docs for Swagger UI."})

    @app.get("/style.css", include_in_schema=False)
    def serve_style_css():
        return FileResponse(str(PUBLIC_DIR / "style.css"))

    @app.get("/app.js", include_in_schema=False)
    def serve_app_js():
        return FileResponse(str(PUBLIC_DIR / "app.js"))

