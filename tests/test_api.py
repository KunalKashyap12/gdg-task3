"""
Integration tests for FastAPI endpoints, request schemas, and static file serving.
"""

import unittest
from fastapi.testclient import TestClient
from app import app


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("Sports", data["categories"])

    def test_vercel_api_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    def test_frontend_root_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("News Category Predictor", res.text)

    def test_static_assets_accessible(self):
        for path in ["/static/style.css", "/static/app.js", "/style.css", "/app.js"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"Static asset at {path} failed to load")

    def test_predict_single_article(self):
        payload = {"text": "Manchester United edges Liverpool with decisive second half goal"}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["category_name"], "Sports")
        self.assertEqual(data["category_id"], 2)
        self.assertGreater(data["confidence"], 0.7)

    def test_api_predict_route(self):
        payload = {"text": "SpaceX Starship reaches orbit on historic test mission"}
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["category_name"], "Sci/Tech")

    def test_predict_batch_articles(self):
        payload = {
            "articles": [
                {"title": "Government ministers meet in Geneva for international treaty talks"},
                {"title": "Tech stocks rally as quarterly revenue tops earnings forecasts"}
            ]
        }
        res = self.client.post("/predict/batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["category_name"], "World")
        self.assertEqual(data[1]["category_name"], "Business")

    def test_predict_empty_payload_validation(self):
        res = self.client.post("/predict", json={})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
