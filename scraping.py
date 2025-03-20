import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}
BASE_URL = "https://www.imdb.com"

def scrape_movie_details(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    
    data = {}
    # Titre
    data["title"] = soup.find("span", class_="hero__primary-text").get_text(strip=True)
    
    # Année
    year_block = soup.find("a", class_="ipc-link--baseAlt", href=lambda x: x and "releaseinfo" in x)
    data["year"] = year_block.get_text(strip=True) if year_block else None
    
    # Note
    rating_block = soup.find("div", {"data-testid": "hero-rating-bar__aggregate-rating__score"})
    data["rating"] = rating_block.find("span").get_text(strip=True) if rating_block else None
    
    # Résumé
    summary_block = soup.find("span", {"data-testid": "plot-l"})
    data["summary"] = summary_block.get_text(strip=True) if summary_block else None
    
    return data

def scrape_imdb_top_movies(max_movies=250):
    url = f"{BASE_URL}/chart/top/"
    response = requests.get(url, headers=HEADERS)
    
    # Vérifier si la requête a réussi
    if response.status_code != 200:
        raise Exception(f"Échec de la requête (code {response.status_code})")
    
    soup = BeautifulSoup(response.content, "html.parser")
    movies = []
    
    # Nouveau sélecteur pour les lignes du tableau
    movie_rows = soup.select("ul.ipc-metadata-list li.ipc-metadata-list-summary-item")
    
    for row in movie_rows[:max_movies]:
        link = row.find("a", class_="ipc-title-link-wrapper")["href"]
        movie_url = BASE_URL + link.split("?")[0]
        
        movie_data = scrape_movie_details(movie_url)
        movies.append(movie_data)
        time.sleep(1.5)  # Respecter les politesses de scraping
        title = movie_data["title"]
        print(f"Movie: {title} scraped")
    return pd.DataFrame(movies)

# Exécution
if __name__ == "__main__":
    try:
        df = scrape_imdb_top_movies(max_movies=250)
        df.to_csv("imdb_movies.csv", index=False)
        print("Sauvegarde réussie !")
    except Exception as e:
        print(f"Erreur : {e}")