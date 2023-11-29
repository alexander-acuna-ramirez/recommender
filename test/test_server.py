import pytest
import sys
import os
from flask import Flask
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_path)
from server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_content_based_endpoint_returns_ok_status(client):
    response = client.get('/recomendations/content-based/1?amount=5')
    assert response.status_code == 200

def test_content_based_endpoint_returns_array_of_expected_size(client):
    response = client.get('/recomendations/content-based/1?amount=5')
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 5  # Verifica que el tamaño del array sea el esperado

def test_user_based_endpoint_returns_ok_status(client):
    response = client.get('/recommendations/user-based/1?amount=5')
    assert response.status_code == 200

def test_user_based_endpoint_returns_array_of_expected_size(client):
    response = client.get('/recommendations/user-based/1?amount=5')
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 5  # Verifica que el tamaño del array sea el esperado

def test_hybrid_based_endpoint_returns_ok_status(client):
    response = client.get('/recomendations/hybrid-based/1/2?amount=5')
    assert response.status_code == 200

def test_hybrid_based_endpoint_returns_array_of_expected_size(client):
    response = client.get('/recomendations/hybrid-based/1/2?amount=5')
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 5  # Verifica que el tamaño del array sea el esperado

def test_content_based_not_found(client):
    response = client.get('/recomendations/content-based/1000?amount=5')
    assert response.status_code == 404
    assert response.json == {"error": "Content not found", "message": "Contenido no existente"}

def test_user_based_user_not_found(client):
    response = client.get('/recommendations/user-based/1000')
    assert response.status_code == 404
    assert response.json == {"error": "User not found", "message": "Usuario no existente"}

def test_content_based_invalid_amount(client):
    response = client.get('/recomendations/content-based/1?amount=-5')
    assert response.status_code == 400
    assert response.json == {"error": "Invalid arguments", "message": "Invalid amount argument"}

def test_user_based_invalid_amount(client):
    response = client.get('/recommendations/user-based/1?amount=-5')
    assert response.status_code == 400
    assert response.json == {"error": "Invalid arguments", "message": "Invalid amount argument"}

def test_hybrid_based_invalid_amount(client):
    response = client.get('/recomendations/hybrid-based/1/2?amount=-5')
    assert response.status_code == 400
    assert response.json == {"error": "Invalid arguments", "message": "Invalid amount argument"}


def test_hybrid_based_user_not_found(client):
    response = client.get('/recomendations/hybrid-based/999/2?amount=5')
    assert response.status_code == 404
    assert response.json == {"error": "User not found", "message": "Usuario no existente"}

def test_hybrid_based_content_not_found(client):
    response = client.get('/recomendations/hybrid-based/1/999?amount=5')
    assert response.status_code == 404
    assert response.json == {"error": "Content not found", "message": "Contenido no existente"}
