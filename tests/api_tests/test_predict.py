import unittest
from fastapi.testclient import TestClient
from api.main import app

"""
Unit tests for API prediction endpoint.

All relevant cases for /predict response.
"""

class TestPredictEndpoint(unittest.TestCase):
    """
    Endpoint returns HTTP 200 status code.
    """
    def test_predict_endpoint_returns_200(self):
        client = TestClient(app)
        response = client.get("/predict")
        self.assertEqual(response.status_code, 200, "API did not return 200 status code")

    """
    Response JSON contains the key 'predicted_deals'.
    """
    def test_predict_endpoint_contains_predicted_deals(self):
        client = TestClient(app)
        response = client.get("/predict")
        data = response.json()
        self.assertIn("predicted_deals", data, "Response does not contain predicted_deals key")

    """
    Value for predicted_deals is float or int.
    """
    def test_predicted_deals_is_numeric(self):
        client = TestClient(app)
        response = client.get("/predict")
        data = response.json()
        value = data["predicted_deals"]
        self.assertIsInstance(value, (float, int), "Predicted deals is not numeric")

if __name__ == "__main__":
    unittest.main()
