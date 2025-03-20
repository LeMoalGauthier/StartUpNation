import os
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class MovieApp:
    def __init__(self):
        self.movie_api_key = os.getenv("MOVIE_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.hf_client = InferenceClient(model=self.hf_model, token=self.hf_api_key, timeout=2000)
    
    def get_movies_info(self, movie_title):
        url = f"https://api.themoviedb.org/3/search/movie?api_key={self.movie_api_key}&query={movie_title}&language=fr"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "rating": movie.get("vote_average", "N/A"),
                    "poster_path": movie.get("poster_path", "")
                }
                for movie in data.get("results", [])
            ]
        return None
    
    def generate_prompt(self, movie_title, prompt_type):
        prompts = {
            "poster": (
                f"Génère un descriptif détaillé pour une affiche de film cinématographique du film '{movie_title}'. "
                "Inclut les personnages principaux, l'ambiance, les couleurs dominantes, les éléments visuels clés, "
                "et le style de l'affiche. L'affiche doit avoir un rendu réaliste et spectaculaire, avec un fort impact visuel."
            ),
            "trailer": (
                f"Imagine la bande-annonce du film '{movie_title}'. Décris en détail le ton, la musique, les moments clés, "
                "les dialogues marquants et les effets visuels. La bande-annonce doit être immersive et captivante, "
                "donnant envie de voir le film immédiatement."
            )
        }
        response = self.hf_client.text_generation(prompts[prompt_type])
        return response if response else f"Impossible de générer un prompt pour {prompt_type}."
    
    def get_movie_details(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={self.movie_api_key}&language=fr"
        response = requests.get(url)
        
        if response.status_code == 200:
            movie_details = response.json()
            movie_details['ai_description_for_poster'] = self.generate_prompt(movie_details['title'], "poster")
            movie_details['ai_description_for_trailer'] = self.generate_prompt(movie_details['title'], "trailer")
            return movie_details
        return None