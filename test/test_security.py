import json
import pytest
import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_path)
from server import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_token_valid_credentials(client):
    response = client.post('/generate-token', json={"usuario": "ceciac", "contraseña": "$2b$12$1l5/ZOPJK6Hs4LdH2vciDej6opEauRkOMIH47uB3LYWJd.4ZnyERS"})
    assert response.status_code == 200
    assert 'access_token' in json.loads(response.data)

def test_api_does_not_returns_data_without_token(client):
    response = client.get('/recomendations/content-based/1?amount=5')
    assert response.status_code == 401

def test_api_does_not_return_data_with_invalid_token(client):
    invalid_token = "12354879887"
    response = client.get('/recomendations/content-based/1?amount=5', headers={'Authorization': f'Bearer {invalid_token}'})
    assert response.status_code == 422

def test_api_returns_content_recommendations_with_valid_token(client):
    response_token = client.post('/generate-token', json={"usuario": "ceciac", "contraseña": "contraseña_prueba"})
    assert response_token.status_code == 200
    valid_token = response_token.json['access_token']
    response_recommendations = client.get('/recomendations/content-based/1?amount=5', headers={'Authorization': f'Bearer {valid_token}'})
    assert response_recommendations.status_code == 200

def test_api_returns_user_recommendations_with_valid_token(client):
    response_token = client.post('/generate-token', json={"usuario": "ceciac", "contraseña": "contraseña_prueba"})
    assert response_token.status_code == 200
    valid_token = response_token.json['access_token']
    response_recommendations = client.get('/recommendations/user-based/1?amount=5', headers={'Authorization': f'Bearer {valid_token}'})
    assert response_recommendations.status_code == 200

def test_api_returns_hybrid_recommendations_with_valid_token(client):
    response_token = client.post('/generate-token', json={"usuario": "ceciac", "contraseña": "contraseña_prueba"})
    assert response_token.status_code == 200
    valid_token = response_token.json['access_token']
    response_recommendations = client.get('/recomendations/hybrid-based/1/1?amount=5', headers={'Authorization': f'Bearer {valid_token}'})
    assert response_recommendations.status_code == 200



# Otros tests pueden incluir casos para manejar excepciones, configuraciones incorrectas, etc.
