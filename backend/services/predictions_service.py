import joblib
import numpy as np

class PredictionService:
    def __init__(self, model_path='models/best_model.joblib'):
        self.model = joblib.load(model_path)

    def predict(self, features: list) -> int:
        input_array = np.array([features])
        prediction = self.model.predict(input_array)
        return int(prediction[0])
