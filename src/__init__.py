"""
News Classification System Package.
A modular NLP pipeline for classifying news articles into World, Sports, Business, and Sci/Tech.
"""

from src.predict import NewsClassifier, predict_article
from src.preprocessing import TextPreprocessor
from src.pipeline import run_pipeline
from src.config import LABEL_MAP, CLASS_NAMES

__version__ = "1.0.0"
__all__ = [
    "NewsClassifier",
    "predict_article",
    "TextPreprocessor",
    "run_pipeline",
    "LABEL_MAP",
    "CLASS_NAMES"
]

