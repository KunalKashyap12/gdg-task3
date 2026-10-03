"""
Model definition and factory module for News Classification System.
Includes Logistic Regression, LinearSVC, Calibrated Classifiers,
and Ensemble models.
"""

from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

from src.config import RANDOM_STATE
from src.features import build_feature_pipeline


def get_available_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Returns a dictionary of candidate NLP classification models.
    Keeps the two primary algorithms: LinearSVC and LogisticRegression.
    """
    models = {
        "LinearSVC": LinearSVC(
            C=0.5,
            loss="squared_hinge",
            penalty="l2",
            dual="auto",
            random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(
            C=2.0,
            max_iter=300,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def create_classification_pipeline(
    model: Optional[BaseEstimator] = None,
    include_char_ngrams: bool = False,
    include_metadata: bool = False,
    word_max_features: int = 35000,
) -> Pipeline:
    """
    Creates an end-to-end scikit-learn Pipeline with feature extraction and classifier.
    """
    if model is None:
        # Default high-performance calibrated classifier
        model = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=0.5,
                loss="squared_hinge",
                penalty="l2",
                dual="auto",
                random_state=RANDOM_STATE
            ),
            method="sigmoid",
            cv=3
        )

    features = build_feature_pipeline(
        include_char_ngrams=include_char_ngrams,
        include_metadata=include_metadata,
        word_max_features=word_max_features
    )

    pipeline = Pipeline([
        ("features", features),
        ("classifier", model)
    ])

    return pipeline


def build_voting_ensemble(random_state: int = RANDOM_STATE) -> VotingClassifier:
    """
    Constructs a soft-voting ensemble combining Logistic Regression
    and Calibrated LinearSVC.
    """
    clf1 = LogisticRegression(C=2.0, max_iter=300, random_state=random_state)
    clf2 = CalibratedClassifierCV(
        estimator=LinearSVC(C=0.5, random_state=random_state, dual="auto"),
        method="sigmoid",
        cv=3
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", clf1),
            ("svc", clf2)
        ],
        voting="soft",
        weights=[1.0, 1.5]
    )
    return ensemble
