"""
Unit tests for model inference engine and prediction output structure.
"""

import unittest
import pandas as pd
from src.predict import NewsClassifier, predict_article
from src.config import CLASS_NAMES


class TestNewsClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = NewsClassifier()

    def test_pipeline_loaded(self):
        self.assertIsNotNone(self.classifier.pipeline, "Pipeline failed to load")

    def test_predict_single_article(self):
        result = self.classifier.predict(
            title="Manchester United wins Premier League derby",
            description="Late goal in the second half seals victory for the home team."
        )

        self.assertIn("category_id", result)
        self.assertIn("category_name", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

        self.assertEqual(result["category_name"], "Sports")
        self.assertEqual(result["category_id"], 2)
        self.assertGreaterEqual(result["confidence"], 0.5)

        # Check all 4 category probabilities exist and sum close to 1.0
        for name in CLASS_NAMES:
            self.assertIn(name, result["probabilities"])
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_batch(self):
        df = pd.DataFrame([
            {"Title": "Apple announces new iPhone with M4 chip", "Description": "The device features enhanced neural processing."},
            {"Title": "Central bank raises benchmark interest rate", "Description": "Policymakers target sticky inflation amid market volatility."}
        ])

        results_df = self.classifier.predict_batch(df)
        self.assertIn("pred_category", results_df.columns)
        self.assertIn("confidence", results_df.columns)
        self.assertEqual(len(results_df), 2)


if __name__ == "__main__":
    unittest.main()
