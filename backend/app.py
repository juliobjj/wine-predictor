import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

CORS(app)
model = joblib.load('models/best_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)
    features = np.array([data['features']])  # Lista de 11 valores numéricos
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    print("Executando testes com Pytest...")
    result = subprocess.run(["pytest", "-s", "tests/test_model_quality.py"])

    if result.returncode != 0:
        print("Testes falharam. Interrompendo execução do app.")
        exit(1)  # Encerra com erro

    print("Testes passaram com sucesso. Iniciando a aplicação...")
    app.run(debug=True)
