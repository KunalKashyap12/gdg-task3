"""
Pydantic data validation schemas for the News Classification API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SingleArticleRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="The news article title or headline.",
        example="NASA James Webb Telescope captures deepest cosmic infrared view"
    )
    description: Optional[str] = Field(
        default="",
        description="Optional article body or subtitle context.",
        example="The observatory revealed galaxies formed mere hundreds of millions of years after the Big Bang."
    )
    text: Optional[str] = Field(
        default=None,
        description="Raw full news article text or headline.",
        example="NASA James Webb Telescope captures deepest cosmic infrared view. The observatory revealed galaxies formed after the Big Bang."
    )


class BatchArticleRequest(BaseModel):
    articles: List[SingleArticleRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of articles to classify in a single batch request."
    )


class SingleArticleResponse(BaseModel):
    title: str
    description: str
    category_id: int = Field(..., description="Category index (1: World, 2: Sports, 3: Business, 4: Sci/Tech)")
    category_name: str = Field(..., description="Human-readable category label")
    confidence: float = Field(..., description="Prediction probability confidence between 0.0 and 1.0")
    probabilities: Dict[str, float] = Field(..., description="Calibrated probabilities across all 4 categories")


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    categories: List[str]
