"""
Feature extraction module for News Classification System.
Supports TF-IDF (word & char n-grams) and engineered text metadata features.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_SUBLINEAR_TF,
)


class TextMetadataExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numerical metadata features from raw text:
    - Character length
    - Word count
    - Average word length
    - Ratio of uppercase characters
    - Ratio of digits (financial figures, sports scores, stats)
    - Punctuation density
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str) or len(text) == 0:
                features.append([0, 0, 0.0, 0.0, 0.0, 0.0])
                continue

            char_len = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_len = char_len / max(word_count, 1)

            upper_count = sum(1 for c in text if c.isupper())
            upper_ratio = upper_count / max(char_len, 1)

            digit_count = sum(1 for c in text if c.isdigit())
            digit_ratio = digit_count / max(char_len, 1)

            punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            punct_ratio = punct_count / max(char_len, 1)

            features.append([
                char_len,
                word_count,
                avg_word_len,
                upper_ratio,
                digit_ratio,
                punct_ratio
            ])

        return np.array(features, dtype=np.float32)


def get_word_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
    stop_words: Optional[str] = "english",
) -> TfidfVectorizer:
    """
    Constructs a word-level TF-IDF vectorizer with unigrams and bigrams.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
        strip_accents="unicode",
        norm="l2",
    )


def get_char_tfidf_vectorizer(
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 3,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Constructs a character n-gram TF-IDF vectorizer for subword representations.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        norm="l2",
    )


def build_feature_pipeline(
    include_char_ngrams: bool = False,
    include_metadata: bool = False,
    word_max_features: int = TFIDF_MAX_FEATURES,
    word_ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> BaseEstimator:
    """
    Builds a flexible feature extraction pipeline.
    If only word TF-IDF is requested, returns TfidfVectorizer directly.
    Otherwise, builds a FeatureUnion combining requested extractors.
    """
    if not include_char_ngrams and not include_metadata:
        return get_word_tfidf_vectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range
        )

    transformer_list = [
        ("word_tfidf", get_word_tfidf_vectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range
        ))
    ]

    if include_char_ngrams:
        transformer_list.append((
            "char_tfidf",
            get_char_tfidf_vectorizer(max_features=10000)
        ))

    if include_metadata:
        transformer_list.append((
            "metadata",
            Pipeline([
                ("extractor", TextMetadataExtractor()),
                ("scaler", StandardScaler(with_mean=False))
            ])
        ))

    return FeatureUnion(transformer_list=transformer_list)
