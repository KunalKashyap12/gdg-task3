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
        # Champion pipeline was trained on clean_basic; avoid heavy NLTK downloads on serverless cold starts
        self.preprocessor = TextPreprocessor(use_lemmatization=False, remove_stopwords=False)
        self.pipeline = None
        self.load_error: Optional[str] = None

        # Search across potential deployment locations
        candidate_paths = [
            self.model_path,
            Path(__file__).resolve().parent.parent / "models" / "news_classifier_pipeline.pkl",
            Path.cwd() / "models" / "news_classifier_pipeline.pkl",
            Path("/var/task/models/news_classifier_pipeline.pkl"),
        ]

        found_path = None
        for p in candidate_paths:
            if p and p.is_file():
                found_path = p
                break

        if found_path:
            self.load_model(found_path)
        else:
            searched = [str(p) for p in candidate_paths]
            self.load_error = f"Model artifact not found. Checked: {searched}"
            print(f"[Warning] {self.load_error}")

    def load_model(self, model_path: Union[str, Path]):
        """
        Loads the trained model pipeline artifact from disk using pickle safely.
        """
        self.model_path = Path(model_path)
        try:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            self.load_error = None
            print(f"Loaded news classification pipeline from {self.model_path}")
        except Exception as e:
            self.pipeline = None
            self.load_error = f"Failed to unpickle model from {self.model_path}: {str(e)}"
            print(f"[Error] {self.load_error}")

    def predict(self, title: str, description: str = "") -> Dict[str, Any]:
        """
        Classifies a single news article given its title and description.
        Returns:
            Dict containing predicted category_id, category_name, confidence,
            and probabilities for all 4 categories.
        """
        if self.pipeline is None:
            raise RuntimeError(self.load_error or "Model pipeline is not loaded.")

        # Combine and clean text
        raw_combined = self.preprocessor.combine_title_description(title, description)
        cleaned_text = self.preprocessor.clean_basic(raw_combined)

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
