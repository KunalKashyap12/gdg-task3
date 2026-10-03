"""
Configuration module for News Classification System.
Defines paths, label mappings, hyperparameters, and regex patterns.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = MODELS_DIR  # Alias for backward compatibility

# Data file paths (resolves in data/ or root)
TRAIN_DATA_PATH = DATA_DIR / "train.csv" if (DATA_DIR / "train.csv").exists() else PROJECT_ROOT / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv" if (DATA_DIR / "test.csv").exists() else PROJECT_ROOT / "test.csv"

# Model file paths
MODEL_PATH = MODELS_DIR / "news_classifier_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
PER_CLASS_METRICS_PATH = MODELS_DIR / "per_class_metrics.png"
TOP_WORDS_PATH = MODELS_DIR / "top_keywords_per_category.png"

# Class Mapping: IDs 1-4
LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]

# Color palette for consistent visualization
CLASS_COLORS = {
    "World": "#3B82F6",     # Blue
    "Sports": "#10B981",    # Emerald Green
    "Business": "#F59E0B",  # Amber Gold
    "Sci/Tech": "#8B5CF6"   # Purple
}

# Common English Contractions for Text Normalization
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "u.s.": "usa",
    "u.k.": "uk",
    "e.u.": "eu",
    "i.e.": "that is",
    "e.g.": "for example"
}

# Common news agency boilerplate prefixes/suffixes to strip
NEWS_AGENCY_PATTERNS = [
    r"^\s*(reuters|ap|afp|bloomberg|xinhua|pr\s*newswire|business\s*wire)\s*[-—–:]\s*",
    r"\s*\((reuters|ap|afp|bloomberg|space\.com|cnet|zdnet|marketwatch|financial\s*times)\)\s*",
    r"\b(reuters|afp|bloomberg)\b\s*[-—–:]\s*"
]

# Random State
RANDOM_STATE = 42

# TF-IDF Feature Extraction Hyperparameters
TFIDF_MAX_FEATURES = 35000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and Bigrams
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True
