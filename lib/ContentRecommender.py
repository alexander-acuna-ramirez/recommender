import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import unicodedata
import re  # Importar el módulo re para trabajar con expresiones regulares
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .Exceptions.ContentNotFoundException import ContentNotFoundException

# Implementamos la clase ContentRecommenderModule que hereda de RecommenderModule
class ContentRecommender():
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))
        self.data = None
        self.indices = None
        self.tfidf_matrix = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='|')
        finaldata = data[["title", "description", "category", "content_type", "tags", "Id"]]
        finaldata['name'] = finaldata['title'].copy()
        finaldata = finaldata.set_index('Id')
        finaldata = finaldata.apply(self.preprocess_columns, axis=1)
        finaldata['preprocessed_data'] = finaldata[['preprocessed_title', 'preprocessed_description', 'preprocessed_tags']].apply(lambda row: ' '.join(row), axis=1)
        self.data = finaldata
        self.indices = pd.Series(finaldata.index)
        self.tfidf_matrix = self.calculate_tfidf_matrix()

    """
    def recommend(self, id, num_recommendations=5):
        try:
            index = self.indices[self.indices == id].index[0]
            similarity_scores = pd.Series(cosine_similarity(self.tfidf_matrix[index].reshape(1, -1), self.tfidf_matrix).flatten()).sort_values(ascending=False)
            recommended_items = [(list(self.data.index)[i], similarity_scores[i]) for i in range(1, num_recommendations + 1)]
            return recommended_items
        except IndexError:
            raise ContentNotFoundException
    """
    
    def recommend(self, id, num_recommendations=5):
        try:
            index = self.indices[self.indices == id].index[0]
            similarity_scores = pd.Series(cosine_similarity(self.tfidf_matrix[index].reshape(1, -1), self.tfidf_matrix).flatten())
            recommended_items = sorted([(list(self.data.index)[i], similarity_scores[i]) for i in range(len(similarity_scores))], key=lambda x: x[1], reverse=True)[:num_recommendations]  # Ordenar por puntaje descendente y seleccionar las primeras n recomendaciones
            return recommended_items
        except IndexError:
            raise ContentNotFoundException

    
    def recommend_all(self, id):
        try:
            index = self.indices[self.indices == id].index[0]
            similarity_scores = pd.Series(cosine_similarity(self.tfidf_matrix[index].reshape(1, -1), self.tfidf_matrix).flatten()).sort_values(ascending=False)
            recommended_items = [(list(self.data.index)[i], similarity_scores[i]) for i in range(len(similarity_scores))]
            return recommended_items
        except IndexError:
            raise ContentNotFoundException
    
    def get_features(self, user_id):
        return self.tfidf_matrix
    
    @staticmethod
    def remove_symbols(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def preprocess_columns(self, row):
        # Preprocesar la descripción
        description = row["description"]
        description = ''.join((c for c in unicodedata.normalize('NFD', description) if unicodedata.category(c) != 'Mn'))
        description = description.lower()
        description = nltk.word_tokenize(description, language='spanish')
        description = [self.lemmatizer.lemmatize(word) for word in description if word not in self.stop_words]
        description = [self.remove_symbols(word) for word in description]  # Eliminar símbolos
        row["preprocessed_description"] = ' '.join(description)
        # Preprocesar el título
        title = row["name"]
        title = ''.join((c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn'))
        title = title.lower()
        title = nltk.word_tokenize(title, language='spanish')
        title = [self.lemmatizer.lemmatize(word) for word in title if word not in self.stop_words]
        title = [self.remove_symbols(word) for word in title]  # Eliminar símbolos
        row["preprocessed_title"] = ' '.join(title)
        # Preprocesar las etiquetas (tags)
        tags = row["tags"]
        tags = tags.split(",")  # Suponiendo que las etiquetas están separadas por comas
        tags = [tag.strip().lower() for tag in tags]
        tags = [self.remove_symbols(tag) for tag in tags]  # Eliminar símbolos
        row["preprocessed_tags"] = ', '.join(tags)
        # Preprocesar la categoría
        category = row["category"]
        category = ''.join((c for c in unicodedata.normalize('NFD', category) if unicodedata.category(c) != 'Mn')).lower()
        category = self.remove_symbols(category)  # Eliminar símbolos
        row["preprocessed_category"] = category
        # Tipo de contenido (content_type) no necesita preprocesamiento
        return row

    def calculate_tfidf_matrix(self):
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.data["preprocessed_data"])
        return tfidf_matrix

"""
content_recommender = ContentRecommender()
content_recommender.load_data("../data/data.csv")
print(content_recommender.recommend(1, 50))
"""