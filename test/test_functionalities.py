import requests

def test_content_based_recommendation():
    base_url = "http://127.0.0.1:5000"
    content_id = 1
    response = requests.get(f"{base_url}/recomendations/content-based/{content_id}?amount=5")
    assert response.status_code == 200
    recommendations = response.json()
    assert isinstance(recommendations, list)
    assert len(recommendations) == 5


def test_user_based_recommendation():
    base_url = "http://127.0.0.1:5000"
    user_id = 1
    response = requests.get(f"{base_url}/recomendations/content-based/{user_id}?amount=5")
    assert response.status_code == 200
    recommendations = response.json()
    assert isinstance(recommendations, list)
    assert len(recommendations) == 5

def test_hybrid_based_recommendation():
    base_url = "http://127.0.0.1:5000"
    content_id = 1
    user_id = 1
    response = requests.get(f"{base_url}/recomendations/hybrid-based/{user_id}/{content_id}?amount=5")
    assert response.status_code == 200
    recommendations = response.json()
    assert isinstance(recommendations, list)
    assert len(recommendations) == 5
