import pytest
import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_path)
from lib.HybridRecommender import HybridRecommender
from lib.ContentRecommender import ContentRecommender
from lib.CollaborativeFilteringRecommender import CollaborativeFilteringRecommender
from lib.Exceptions.ContentNotFoundException import ContentNotFoundException
from lib.Exceptions.UserNotFoundException import UserNotFoundException


@pytest.fixture
def hybrid_recommender():
    content_recommender = ContentRecommender()
    content_recommender.load_data("../data/data.csv")
    collaborative_recommender = CollaborativeFilteringRecommender("../data/interactions.csv", 103)
    collaborative_recommender.train_model()
    hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)
    return hybrid_recommender

def test_recommend_n_hybrid_recommendations(hybrid_recommender):
    assert len(hybrid_recommender.recommend_for_user(1, 1, 10) ) == 10

def test_throw_error_on_incorrect_user_id(hybrid_recommender):
    with pytest.raises(UserNotFoundException):
        hybrid_recommender.recommend_for_user(1000, 1, 10)

def test_throw_error_on_incorrect_content_id(hybrid_recommender):
    with pytest.raises(ContentNotFoundException):
        hybrid_recommender.recommend_for_user(1, 10000, 10)




