from flask import Flask, jsonify, request
from lib.ContentRecommender import ContentRecommender
from lib.CollaborativeFilteringRecommender import CollaborativeFilteringRecommender
from lib.HybridRecommender import HybridRecommender
from lib.Exceptions.ContentNotFoundException import ContentNotFoundException
from lib.Exceptions.UserNotFoundException import UserNotFoundException
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from dotenv import load_dotenv
import os
import bcrypt
# Cargar variables de entorno desde el archivo .env
load_dotenv()

import pandas as pd
#Modules start
content_recommender = ContentRecommender()
content_recommender.load_data("../data/data.csv")
collaborative_recommender = CollaborativeFilteringRecommender('../data/interactions.csv', 103)
df = pd.read_csv("../data/data.csv", sep="|")

collaborative_recommender.train_model()
hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)


app = Flask(__name__)

app.config['JWT_SECRET_KEY'] = 'tu_clave_secreta'  # Cambia esto por una clave secreta segura en producción
jwt = JWTManager(app)

# Obtiene el usuario y la contraseña desde el archivo .env
env_usuario = os.getenv('SECRET_USER')
env_contraseña_hashed = os.getenv('SECRET_TOKEN')

# Endpoint para generar un token
@app.route('/generate-token', methods=['POST'])
def generate_token():

        # Asumiendo que recibes el usuario y la contraseña en el cuerpo de la solicitud
    input_usuario = request.json.get('usuario')
    input_contraseña = request.json.get('contraseña')

    if input_usuario == env_usuario:
        access_token = create_access_token(identity=input_usuario)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify( { "error": "Credenciales incorrectas"}), 401


@app.route('/recommendations/user-based/<int:user_id>', methods=['GET'])
@jwt_required()
def user_based(user_id):
    try:
        amount = int(request.args.get('amount', 5))
        if amount < 0:
            raise ValueError("Invalid amount argument")
        recommendations = collaborative_recommender.recommend_for_user(user_id, amount)

        content_ids = [item[0] for item in recommendations]
        scores = [item[1] for item in recommendations]
        recommendation_list = []
        for content_id, score in zip(content_ids, scores):
            content_info = df[df['Id'] == content_id].iloc[0]
            recommendation_dict = {
                'content_id': content_id,
                'content_name': content_info['title'],
                'recommendation_score': score
            }
            recommendation_list.append(recommendation_dict)
        return jsonify(recommendation_list), 200
        #return jsonify(recommendations), 200
    except UserNotFoundException as e:
        return jsonify({"error": "User not found", "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": "Invalid arguments", "message": str(e)}), 400



@app.route('/recomendations/content-based/<int:content_id>', methods=['GET'])
@jwt_required()
def content_based(content_id):
    try:
        amount = int(request.args.get('amount', 5))
        if amount < 0:
            raise ValueError("Invalid amount argument")
        recommendations = content_recommender.recommend(content_id, amount)
        content_ids = [item[0] for item in recommendations]
        scores = [item[1] for item in recommendations]
        recommendation_list = []
        for content_id, score in zip(content_ids, scores):
            content_info = df[df['Id'] == content_id].iloc[0]
            recommendation_dict = {
                'content_id': content_id,
                'content_name': content_info['title'],
                'recommendation_score': score
            }
            recommendation_list.append(recommendation_dict)
        return jsonify(recommendation_list), 200
    except ContentNotFoundException as e:
        return jsonify({"error": "Content not found", "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": "Invalid arguments", "message": str(e)}), 400
        

@app.route('/recomendations/hybrid-based/<int:user_id>/<int:content_id>', methods=['GET'])
@jwt_required()
def hybrid_based(user_id, content_id):
    try:
        amount = int(request.args.get('amount', 5))
        if amount < 0:
            raise ValueError("Invalid amount argument")
        recommendations = hybrid_recommender.recommend_for_user(user_id, content_id, amount)
        content_ids = [item[0] for item in recommendations]
        scores = [item[1] for item in recommendations]
        recommendation_list = []
        for content_id, score in zip(content_ids, scores):
            content_info = df[df['Id'] == content_id].iloc[0]
            recommendation_dict = {
                'content_id': content_id,
                'content_name': content_info['title'],
                'recommendation_score': score
            }
            recommendation_list.append(recommendation_dict)
        return jsonify(recommendation_list), 200
    except UserNotFoundException as e:
        return jsonify({"error": "User not found", "message": str(e)}), 404
    except ContentNotFoundException as e:
        return jsonify({"error": "Content not found", "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": "Invalid arguments", "message": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True ,threaded=True)