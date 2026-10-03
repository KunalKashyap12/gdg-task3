"""
Inference and prediction module for News Classification System.
Provides easy-to-use classes and functions to classify single news articles or batches.
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import pickle
import pandas as pd
import numpy as np

from src.config import MODEL_PATH, LABEL_MAP, INV_LABEL_MAP, CLASS_NAMES
from src.preprocessing import TextPreprocessor


class NewsClassifier:
    """
    Inference engine that wraps preprocessing, feature extraction, and model prediction.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
        self.pipeline = None

        if self.model_path.exists():
            self.load_model(self.model_path)

    def load_model(self, model_path: Union[str, Path]):
        """
        Loads the trained model pipeline artifact from disk using pickle.
        """
        self.model_path = Path(model_path)
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"Loaded news classification pipeline from {self.model_path}")

    def predict(self, title: str, description: str = "") -> Dict[str, Any]:
        """
        Classifies a single news article given its title and description.
        Returns:
            Dict containing predicted category_id, category_name, confidence,
            and probabilities for all 4 categories.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded. Train a model or provide valid model_path.")

        # Combine and clean text
        raw_combined = self.preprocessor.combine_title_description(title, description)
        cleaned_text = self.preprocessor.clean_advanced(raw_combined)

        # Predict
        pred_label = self.pipeline.predict([cleaned_text])[0]

        # Get probabilities if supported
        probabilities = {}
        confidence = 1.0
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            # Pipeline classes_ may be [1, 2, 3, 4]
            classes = self.pipeline.classes_
            for cls_id, prob in zip(classes, probs):
                probabilities[LABEL_MAP[cls_id]] = round(float(prob), 4)
            confidence = round(float(np.max(probs)), 4)
        elif hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function([cleaned_text])[0]
            # Softmax approximation
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / np.sum(exp_d)
            for cls_id, prob in zip(self.pipeline.classes_, probs):
                probabilities[LABEL_MAP[cls_id]] = round(float(prob), 4)
            confidence = round(float(np.max(probs)), 4)

        return {
            "title": title,
            "description": description,
            "category_id": int(pred_label),
            "category_name": LABEL_MAP[pred_label],
            "confidence": confidence,
            "probabilities": probabilities
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies a batch of news articles from a DataFrame with 'Title' and 'Description'.
        Returns a DataFrame augmented with predictions.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        result_df = df.copy()
        cleaned_texts = self.preprocessor.process_dataframe(result_df, advanced=True)

        preds = self.pipeline.predict(cleaned_texts)
        result_df["pred_class_id"] = preds
        result_df["pred_category"] = [LABEL_MAP[p] for p in preds]

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(cleaned_texts)
            for idx, cls_id in enumerate(self.pipeline.classes_):
                result_df[f"prob_{LABEL_MAP[cls_id].lower()}"] = np.round(probs[:, idx], 4)
            result_df["confidence"] = np.round(np.max(probs, axis=1), 4)

        return result_df


def predict_article(
    title: str,
    description: str = "",
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function to classify an article in one call.
    """
    classifier = NewsClassifier(model_path=model_path)
    return classifier.predict(title, description)
