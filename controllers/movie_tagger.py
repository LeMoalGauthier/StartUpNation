import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import json
from transformers import pipeline

# Télécharger les ressources NLTK nécessaires
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Charger le dataset
def load_movie_data(csv_path):
    movies_df = pd.read_csv(csv_path)
    return movies_df

# Prétraitement des données
def preprocess_data(df):
    # Convertir les durées en numérique et gérer les valeurs manquantes
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    
    # Gérer les valeurs manquantes dans les colonnes de type texte
    text_columns = ['summary', 'genres', 'directors', 'actors', 'nationality']
    for col in text_columns:
        df[col] = df[col].fillna('')
    
    # Convertir les notes en numérique
    rating_columns = ['press_rating', 'spec_rating']
    for col in rating_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Fonction pour générer des tags automatiques
def generate_tags(movies_df):
    # Initialiser l'analyseur de sentiment
    sia = SentimentIntensityAnalyzer()
    
    # Initialiser le pipeline de zero-shot classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Définir les catégories possibles pour chaque type de tag
    tone_categories = ["comique", "dramatique", "sérieux", "léger", "sombre", "satirique", "ironique", "absurde", "poétique", "intense"]  
    mood_categories = ["optimiste", "pessimiste", "nostalgique", "mélancolique", "romantique", "angoissant", "énergique", "rêveur", "épique", "dérangeant"]  
    theme_categories = ["amour", "vengeance", "rédemption", "trahison", "pouvoir", "survie", "famille", "amitié", "quête", "justice", "guerre", "liberté"]  

    
    # Initialiser les colonnes de tags
    movies_df['sentiment_score'] = None
    movies_df['sentiment_tag'] = None
    movies_df['tone_tags'] = None
    movies_df['mood_tags'] = None
    movies_df['theme_tags'] = None
    movies_df['all_tags'] = None
    
    for idx, row in movies_df.iterrows():
        text_to_analyze = f"{row['title']} {row['summary']} {row['genres']}"
        
        # Analyser le sentiment
        sentiment = sia.polarity_scores(text_to_analyze)
        sentiment_score = sentiment['compound']
        movies_df.at[idx, 'sentiment_score'] = sentiment_score
        
        # Assigner un tag de sentiment
        if sentiment_score >= 0.2:
            movies_df.at[idx, 'sentiment_tag'] = "positif"
        elif sentiment_score <= -0.2:
            movies_df.at[idx, 'sentiment_tag'] = "négatif"
        else:
            movies_df.at[idx, 'sentiment_tag'] = "neutre"
        
        # Classifier le ton, l'ambiance et les thèmes si le résumé n'est pas vide
        if row['summary'] and len(row['summary']) > 10:
            # Classifier le ton
            tone_result = classifier(row['summary'], tone_categories, multi_label=True)
            tone_tags = [tone_result['labels'][i] for i in range(min(3, len(tone_result['labels']))) 
                         if tone_result['scores'][i] > 0.3]
            movies_df.at[idx, 'tone_tags'] = ','.join(tone_tags) if tone_tags else ''
            
            # Classifier l'ambiance
            mood_result = classifier(row['summary'], mood_categories, multi_label=True)
            mood_tags = [mood_result['labels'][i] for i in range(min(3, len(mood_result['labels']))) 
                         if mood_result['scores'][i] > 0.3]
            movies_df.at[idx, 'mood_tags'] = ','.join(mood_tags) if mood_tags else ''
            
            # Classifier les thèmes
            theme_result = classifier(row['summary'], theme_categories, multi_label=True)
            theme_tags = [theme_result['labels'][i] for i in range(min(3, len(theme_result['labels']))) 
                          if theme_result['scores'][i] > 0.3]
            movies_df.at[idx, 'theme_tags'] = ','.join(theme_tags) if theme_tags else ''
        
        # Ajouter les genres comme tags
        genres = row['genres'].split(',') if isinstance(row['genres'], str) else []
        genres = [g.strip() for g in genres]
        
        # Combiner tous les tags
        all_tags = genres.copy()
        
        if isinstance(row['tone_tags'], str) and row['tone_tags']:
            all_tags.extend(row['tone_tags'].split(','))
        
        if isinstance(row['mood_tags'], str) and row['mood_tags']:
            all_tags.extend(row['mood_tags'].split(','))
        
        if isinstance(row['theme_tags'], str) and row['theme_tags']:
            all_tags.extend(row['theme_tags'].split(','))
        
        if row['sentiment_tag']:
            all_tags.append(row['sentiment_tag'])
        
        # Éliminer les doublons et joindre
        all_tags = list(set(all_tags))
        movies_df.at[idx, 'all_tags'] = ','.join(all_tags)
    
    return movies_df

# Fonction pour trouver des films similaires
def find_similar_movies(movies_df):
    # Créer une matrice TF-IDF à partir des résumés et tags
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combiner résumé, tags et genres pour la recherche de similarité
    movies_df['features'] = movies_df['summary'] + ' ' + movies_df['all_tags'] + ' ' + movies_df['genres']
    
    # Remplacer les valeurs NaN par des chaînes vides
    movies_df['features'] = movies_df['features'].fillna('')
    
    # Calculer la matrice TF-IDF
    tfidf_matrix = tfidf.fit_transform(movies_df['features'])
    
    # Calculer la similarité cosinus
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Initialiser une colonne pour les recommandations
    movies_df['recommendations'] = None
    
    # Pour chaque film, trouver les 3 films les plus similaires
    for idx in range(len(movies_df)):
        # Obtenir les scores de similarité pour le film actuel
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Trier les films par score de similarité
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Obtenir les 4 films les plus similaires (y compris le film lui-même)
        sim_scores = sim_scores[1:4]  # Exclure le film lui-même
        
        # Obtenir les indices des films
        movie_indices = [i[0] for i in sim_scores]
        
        # Stocker les IDs des films recommandés
        recommended_ids = movies_df.iloc[movie_indices]['title'].tolist()
        movies_df.at[idx, 'recommendations'] = json.dumps(recommended_ids)
    
    return movies_df

# Fonction principale
def main(csv_path, output_path):
    # Charger les données
    print("Chargement des données...")
    movies_df = load_movie_data(csv_path)
    
    # Prétraiter les données
    print("Prétraitement des données...")
    movies_df = preprocess_data(movies_df)
    
    # Générer des tags
    print("Génération des tags automatiques...")
    movies_df = generate_tags(movies_df)
    
    # Trouver des films similaires
    print("Recherche de films similaires...")
    movies_df = find_similar_movies(movies_df)
    
    # Sauvegarder les résultats
    print(f"Sauvegarde des résultats dans {output_path}...")
    movies_df.to_csv(output_path, index=False)
    
    print("Terminé!")
    return movies_df

if __name__ == "__main__":
    # Paramètres
    input_csv = "projet_cine/allocine_movies.csv"  # Chemin vers le fichier CSV d'entrée
    output_csv = "projet_cine/allocine_movies_enriched.csv"  # Chemin vers le fichier CSV de sortie
    
    # Exécuter le programme
    enriched_df = main(input_csv, output_csv)
    
    # Afficher un exemple de résultat
    print("\nExemple de résultat:")
    sample = enriched_df.iloc[0]
    print(f"Film: {sample['title']}")
    print(f"Tags générés: {sample['all_tags']}")
    print(f"Films recommandés: {sample['recommendations']}")