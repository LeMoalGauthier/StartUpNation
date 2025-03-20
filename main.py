import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True

def get_movie_data(path: str = "imdb_top_1000.csv", title=None) -> dict:
    """Récupère les données d'un film aléatoire avec gestion d'erreur améliorée"""
    try:
        df = pd.read_csv(path, sep=",").dropna(subset=['Overview', 'Genre', 'Series_Title'])
        if title:
            movie = df[df['Series_Title'] == title].iloc[0]
        else:
            movie = df.sample(n=1).iloc[0]
        return {
            "Series_Title": movie["Series_Title"],
            "Genre": movie["Genre"],
            "Overview": movie["Overview"][:200]  # Limite la longueur
        }
    except Exception as e:
        print(f"Erreur de chargement des données: {e}")
        return default_movie_data()

def default_movie_data() -> dict:
    """Données par défaut pour éviter les erreurs"""
    return {
        "Series_Title": "Unknown Movie",
        "Genre": "Drama",
        "Overview": "A mysterious story unfolds in an unknown world."
    }

def improve_prompt(movie_data: dict) -> str:
    """Génère un prompt optimisé pour la génération d'image avec Stable Diffusion."""
    if not movie_data or not all(k in movie_data for k in ("Series_Title", "Genre", "Overview")):
        print("Données de film invalides, utilisation d'un prompt par défaut.")
        return "A visually striking movie poster with a mysterious and cinematic atmosphere."
    
    base_prompt = (
        f"A {movie_data['Genre'].lower()} movie poster titled '{movie_data['Series_Title']}', "
        f"inspired by: {movie_data['Overview']}. Cinematic, high-quality, artistic."
    )
    
    return base_prompt

def generate_img_v2_1(prompt: str, output_path: str = "poster_v2_1.png"):
    """Génération d'image avec optimisations mémoire"""
    try:
        # Configuration du pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Optimisations mémoire
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_sequential_cpu_offload()  # Gestion mémoire avancée
        
        # Génération avec paramètres optimisés
        return pipe(
            prompt=prompt,
            negative_prompt="text, watermark, lowres, bad anatomy",
            width=1024,  # Taille réduite mais viable
            height=1024,
            num_inference_steps=25,
            guidance_scale=9
        ).images[0].save(output_path)
        
    except Exception as e:
        print(f"Erreur critique: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

def generate_img_v1_5(prompt: str, output_path: str = "poster_v1_5.png"):
    """Génération d'image avec optimisations mémoire"""
    try:
        # Configuration du pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Optimisations mémoire
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_sequential_cpu_offload()  # Gestion mémoire avancée
        
        # Génération avec paramètres optimisés
        return pipe(
            prompt=prompt,
            negative_prompt="text, watermark, lowres, bad anatomy",
            width=1024,  # Taille réduite mais viable
            height=1024,
            num_inference_steps=25,
            guidance_scale=9
        ).images[0].save(output_path)
        
    except Exception as e:
        print(f"Erreur critique: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

def get_unique_folder_name(base_name):
    """Génère un nom de dossier unique en ajoutant un numéro s'il existe déjà."""
    if not os.path.exists(base_name):
        return base_name

    index = 2
    while os.path.exists(f"{base_name}({index})"):
        index += 1

    return f"outputs/{base_name}({index})"

def pipeline(movie_data=None, title: str = None):
    """Pipeline de génération d'affiches, accepte soit un titre soit un dictionnaire."""
    
    # Si `movie_data` n'est pas valide, récupérer les données depuis le CSV
    if movie_data is None or not all(k in movie_data for k in ("Series_Title", "Genre", "Overview")):
        if title:
            print(f"Recherche du film '{title}' dans la base...")
        else:
            print("Titre manquant, choix d'un film aléatoire.")
        movie_data = get_movie_data(title=title)
    
    print(f"Movie data: {movie_data}")

    # Créer un dossier unique pour le film
    serie_title = get_unique_folder_name(movie_data["Series_Title"])
    os.makedirs(f"outputs/{serie_title}", exist_ok=True)

    # Génération du prompt
    final_prompt = improve_prompt(movie_data)
    print(f"Final prompt: {final_prompt}")

    # Sauvegarde du prompt
    with open(f"outputs/{serie_title}/prompt", "w") as file:
        file.write(final_prompt)

    # Génération des images
    try:
        generate_img_v2_1(final_prompt, output_path=f"outputs/{serie_title}/poster_v2_1.png")
        generate_img_v1_5(final_prompt, output_path=f"outputs/{serie_title}/poster_v1_5.png")
        print("Génération réussie!")
    except Exception as e:
        print(f"Erreur : {e}")

# Exemples d'utilisation :
if __name__ == "__main__":
    pipeline(title="The Pursuit of Happyness")
    pipeline(movie_data={"Series_Title": "Inception", "Genre": "Sci-Fi", "Overview": "A thief who enters dreams."})
