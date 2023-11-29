import pytest
import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_path)
from lib.ContentRecommender import ContentRecommender
from lib.Exceptions.ContentNotFoundException import ContentNotFoundException
from lib.CollaborativeFilteringRecommender import CollaborativeFilteringRecommender

@pytest.fixture
def content_recommender():
    content_recommender = ContentRecommender()
    content_recommender.load_data("../data/data.csv")
    return content_recommender

def test_recommend_n_elements(content_recommender):
    content_id = 1
    num_recommendations = 5
    recommendations = content_recommender.recommend(content_id, num_recommendations)
    assert len(recommendations) == num_recommendations

def test_recommendations_has_scores(content_recommender):
    user_id = 1
    num_recommendations = 5
    recommendations = content_recommender.recommend(user_id, num_recommendations)
    for _, score in recommendations:
        assert 0 <= score <= 1

def test_all_elements_has_been_scored(content_recommender):
    content_id = 1
    recommendations = content_recommender.recommend_all(content_id)
    assert len(recommendations) == len(content_recommender.indices)

def test_recommender_throws_error_with_not_existing_id(content_recommender):
    content_id = 235
    with pytest.raises(ContentNotFoundException):
        content_recommender.recommend(content_id, num_recommendations=10)

def test_preprocess_columns():
    recommender = ContentRecommender()
    data = {"title": "Ejemplo de título", "description": "Descripción de ejemplo", "tags": "tag1, tag2", "category": "Categoria de Ejemplo", "name": "Ejemplo de nombre", "Id": 1}
    preprocessed_data = recommender.preprocess_columns(data)
    assert "preprocessed_title" in preprocessed_data
    assert "preprocessed_description" in preprocessed_data
    assert "preprocessed_tags" in preprocessed_data
    assert "preprocessed_category" in preprocessed_data
    
