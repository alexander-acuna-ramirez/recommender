from .ContentRecommender import ContentRecommender
from .CollaborativeFilteringRecommender import CollaborativeFilteringRecommender

# ... (resto del código)


class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, content_weight=0.5, collaborative_weight=0.5):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        
    def recommend_for_user(self, user_id, element_id, num_recommendations=10):
        
        content_recommendations = self.content_recommender.recommend_all(element_id)
        collaborative_recommendations = self.collaborative_recommender.recommend_all_for_user(user_id)

        # Escalar los puntajes de Filtrado Colaborativo
        max_collaborative_score = max(collaborative_recommendations, key=lambda x: x[1])[1]
        min_collaborative_score = min(collaborative_recommendations, key=lambda x: x[1])[1]
        scaled_collaborative_recommendations = [(iid, (score - min_collaborative_score) / (max_collaborative_score - min_collaborative_score)) for iid, score in collaborative_recommendations]

        # Ponderar las recomendaciones
        weighted_content_recommendations = [(iid, score * self.content_weight) for iid, score in content_recommendations]
        weighted_collaborative_recommendations = [(iid, score * self.collaborative_weight) for iid, score in scaled_collaborative_recommendations]

        # Combinar las recomendaciones
        all_recommendations = weighted_content_recommendations + weighted_collaborative_recommendations

        # Agrupar las recomendaciones por ítem y sumar las puntuaciones ponderadas
        combined_recommendations = {}
        for iid, score in all_recommendations:
            if iid in combined_recommendations:
                combined_recommendations[iid] += score
            else:
                combined_recommendations[iid] = score

        # Ordenar las recomendaciones por puntuación descendente
        sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)

        # Seleccionar las primeras num_recommendations recomendaciones
        final_recommendations = sorted_recommendations[:num_recommendations]

        return final_recommendations

"""
# Crear instancias de los recomendadores individuales
content_recommender = ContentRecommender()
content_recommender.load_data("../data/data.csv")

collaborative_recommender = CollaborativeFilteringRecommender('../data/interactions.csv', 103)
collaborative_recommender.train_model()

# Crear instancia del recomendador híbrido
hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)

# Obtener recomendaciones para el usuario 1
recommendations = hybrid_recommender.recommend_for_user(user_id=1, element_id=1, num_recommendations=10)
print("Recomendaciones Finales:")
for iid, score in recommendations:
    print(f'Proyecto {iid}: Puntuación = {score}')
"""