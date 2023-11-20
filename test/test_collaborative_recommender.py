import pytest
import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_path)
from lib.CollaborativeFilteringRecommender import CollaborativeFilteringRecommender

@pytest.fixture
def collaborative_recommender():
    file_path = "../data/interactions.csv"  # Reemplaza con la ruta real del archivo de prueba
    num_projects = 103
    collaborative_recommender = CollaborativeFilteringRecommender(file_path, num_projects)
    collaborative_recommender.train_model()
    return collaborative_recommender

def test_recommendation_length(collaborative_recommender):
    user_id = 1
    num_recommendations = 5
    recommendations = collaborative_recommender.recommend_for_user(user_id, num_recommendations)
    assert len(recommendations) == num_recommendations

def test_recommendation_scores(collaborative_recommender):
    user_id = 1
    num_recommendations = 5
    recommendations = collaborative_recommender.recommend_for_user(user_id, num_recommendations)
    for _, score in recommendations:
        assert 1 <= score <= 4

def test_evaluate_model(collaborative_recommender):
    rmse, mae = collaborative_recommender.evaluate_model()
    assert isinstance(rmse, float)
    assert isinstance(mae, float)

def test_load_data():
    recommender = CollaborativeFilteringRecommender("../data/interactions.csv", 103)
    trainset = recommender.load_data()
    assert len(trainset.all_users()) > 0
    assert len(trainset.all_items()) > 0
