import unittest
import numpy as np
from unittest.mock import MagicMock
from main import load_model, get_recommendation, import_and_predict
import test_face

class TestMaskDetection(unittest.TestCase):
    def setUp(self):
        # Load a mock model for testing
        self.model = MagicMock()
        self.result_class = 1  # Assume MASK class for testing

    def test_load_model(self):
        # Test if load_model returns a valid model
        model = load_model()
        self.assertIsNotNone(model)
        # Add more assertions as needed based on your specific use case

    def test_get_recommendation(self):
        # Test get_recommendation for each result class
        self.assertEqual(get_recommendation(0), "Please wear your mask properly.")
        self.assertEqual(get_recommendation(1), "Your mask is detected and worn correctly. Keep wearing it.")
        self.assertEqual(get_recommendation(2), "Please wear a mask for your safety.")
        # Add more assertions as needed based on your specific use case

    def test_import_and_predict(self):
        # Mock image_path and prediction for testing
        image_path = "maksssksksss13.png"
        mock_prediction = np.array([[0.1, 0.8, 0.1]])  # Assume MASK prediction

        self.model.predict.return_value = mock_prediction

        prediction = import_and_predict(image_path, self.model)
        expected_class = np.argmax(mock_prediction)
        self.assertEqual(np.argmax(prediction), expected_class)

if __name__ == '__main__':
    unittest.main()
