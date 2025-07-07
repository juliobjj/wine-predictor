# tests/test_model_quality.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pytest

# Threshold de desempenho mínimo
MIN_F1_SCORE = 0.60

@pytest.fixture
def data():
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    X = df.drop(columns=['quality'])
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_model_f1_score(data):
    X_train, X_test, y_train, y_test = data

    # Carrega o modelo
    model = joblib.load('models/best_model.joblib')

    # Faz predições
    y_pred = model.predict(X_test)

    # Calcula o F1 Score
    score = f1_score(y_test, y_pred, average='weighted')
    print(f"Modelo testado - F1 Score: {score:.4f}")

    # Verifica se passou no teste
    assert score >= MIN_F1_SCORE, f"Modelo falhou: F1 Score {score:.4f} < {MIN_F1_SCORE}"
