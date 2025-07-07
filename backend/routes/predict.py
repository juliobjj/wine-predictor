from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from schemas.wine import WineSampleInput
from database import SessionLocal
from model.wine import WineSample

import joblib
import numpy as np

# Blueprint
router = Blueprint('router', __name__)
model = joblib.load('models/best_model.joblib')

@router.route('/predict', methods=['POST'])
def predict():
    try:
        data = WineSampleInput(**request.get_json())
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400

    features = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    # Criar entrada no banco
    db = SessionLocal()
    try:
        wine = WineSample(
            fixed_acidity=data.features[0],
            volatile_acidity=data.features[1],
            citric_acid=data.features[2],
            residual_sugar=data.features[3],
            chlorides=data.features[4],
            free_sulfur_dioxide=data.features[5],
            total_sulfur_dioxide=data.features[6],
            density=data.features[7],
            ph=data.features[8],
            sulphates=data.features[9],
            alcohol=data.features[10],
            predicted_quality=prediction
        )
        db.add(wine)
        db.commit()
        db.refresh(wine)
    finally:
        db.close()

    return jsonify({'prediction': prediction})

@router.route('/samples', methods=['GET'])
def get_samples():
    db = SessionLocal()
    try:
        samples = db.query(WineSample).all()
        result = [
            {
                'id': s.id,
                'fixed_acidity': s.fixed_acidity,
                'volatile_acidity': s.volatile_acidity,
                'citric_acid': s.citric_acid,
                'residual_sugar': s.residual_sugar,
                'chlorides': s.chlorides,
                'free_sulfur_dioxide': s.free_sulfur_dioxide,
                'total_sulfur_dioxide': s.total_sulfur_dioxide,
                'density': s.density,
                'ph': s.ph,
                'sulphates': s.sulphates,
                'alcohol': s.alcohol,
                'predicted_quality': s.predicted_quality
            }
            for s in samples
        ]
        return jsonify(result)
    finally:
        db.close()
