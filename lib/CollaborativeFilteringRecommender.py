import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from .Exceptions.UserNotFoundException import UserNotFoundException

class CollaborativeFilteringRecommender:
    def __init__(self, file_path, num_projects):
        self.file_path = file_path
        self.num_projects = num_projects
        self.model = None
        self.trainset = None

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df['end_time'] = pd.to_datetime(df['end_time'])
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        df['rating'] = pd.cut(df['duration_seconds'], bins=[0, 5, 60, 300, float('inf')], labels=[1, 2, 3, 4])
        reader = Reader(rating_scale=(1, 4))
        data = Dataset.load_from_df(df[['user_id', 'content_id', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        return trainset

    def train_model(self):
        self.trainset = self.load_data()
        model = SVD()
        model.fit(self.trainset)
        self.model = model

    def evaluate_model(self):
        testset = self.trainset.build_testset()
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')

        return rmse, mae
    
    def recommend_for_user(self, user_id, num_recommendations=10):
        if user_id not in self.trainset.all_users():
            raise UserNotFoundException
        user_items = [item for item in range(1, self.num_projects + 1)]
        user_predictions = [self.model.predict(user_id, item) for item in user_items]
        user_predictions.sort(key=lambda x: x.est, reverse=True)
        top_n = user_predictions[:num_recommendations]
        recommendations = [(prediction.iid, prediction.est) for prediction in top_n]
        return recommendations

    def recommend_all_for_user(self, user_id):
        if user_id not in self.trainset.all_users():
            raise UserNotFoundException
        user_items = [item for item in range(1, self.num_projects + 1)]
        user_predictions = [self.model.predict(user_id, item) for item in user_items]
        user_predictions.sort(key=lambda x: x.est, reverse=True)
        recommendations = [(prediction.iid, prediction.est) for prediction in user_predictions]
        return recommendations

"""
file_path = '../data/interactions.csv'
num_projects = 103
collaborative_recommender = CollaborativeFilteringRecommender(file_path, num_projects)
collaborative_recommender.train_model()
collaborative_recommender.evaluate_model()
collaborative_recommender.recommend_all_for_user(user_id=3423423424321)
"""
