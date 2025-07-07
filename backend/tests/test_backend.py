import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Dados válidos para o modelo
VALID_FEATURES = {
    "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
}

def test_predict_success(client):
    """Teste se a predição retorna sucesso e um valor de qualidade"""
    response = client.post(
        '/predict',
        data=json.dumps(VALID_FEATURES),
        content_type='application/json'
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert isinstance(json_data['prediction'], int)

def test_predict_invalid_input(client):
    """Teste de input inválido para verificar validação"""
    bad_features = {"features": [1, 2]}  # menos de 11 features
    response = client.post(
        '/predict',
        data=json.dumps(bad_features),
        content_type='application/json'
    )
    assert response.status_code == 422 or response.status_code == 400  # dependendo da lib usada
    assert b'error' in response.data or b'message' in response.data

def test_samples_get(client):
    """Teste se o endpoint /samples retorna uma lista de amostras"""
    response = client.get('/samples')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    if data:
        assert 'predicted_quality' in data[0]
