from flask import Blueprint, request, jsonify
from schemas.wine_input import WineInput
from services.predictions_service import PredictionService
from pydantic import ValidationError

predict_route = Blueprint('predict_route', __name__)
predictor = PredictionService()

@predict_route.route('/predict', methods=['POST'])
def predict():
    try:
        data = WineInput(**request.get_json())
        prediction = predictor.predict(data.features)
        return jsonify({'prediction': prediction})
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400
