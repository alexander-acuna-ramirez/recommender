from abc import ABC, abstractmethod

class RecommenderModule(ABC):
    @abstractmethod
    def recommend(self, user_id):
        pass

    @abstractmethod
    def get_features(self, user_id):
        pass
