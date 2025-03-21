#!/usr/bin/env python

from flask import Flask, render_template, request, redirect, url_for
from main import pipeline, get_movie_data
from urllib.parse import unquote
import os
import re  # Ajouter l'importation en haut du fichier
from flask import send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'outputs'

FIXED_CRITERIA = [
    "Net Score",
    "Prompt Adherence", 
    "Visual Quality",
    "Originality",
    "Technical Quality"
]

def safe_folder_name(title):
    """Convertit un titre en nom de dossier sécurisé"""
    # Supprimer les caractères non autorisés dans les noms de fichiers
    title = re.sub(r'[<>:"/\\|?*]', '', title.strip())
    # Remplacer les espaces multiples par un seul espace
    title = re.sub(r'\s+', ' ', title)
    return title

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form.get('title')
        if not title:
            return redirect(url_for('index'))
        return redirect(url_for('generate_poster', title=title))
    return render_template('index.html')

@app.route('/generate/<title>')
def generate_poster(title):
    try:
        # Créer un nom de dossier sécurisé
        safe_title = safe_folder_name(title)
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        # Générer l'affiche avec le titre original
        pipeline(title=title)
        
        # Vérifier si la génération a réussi
        poster_file = os.path.join(output_dir, 'poster_v2_1.png')
        print(f"Looking for: {poster_file}")  # Debugging

        if not os.path.exists(poster_file):
            raise FileNotFoundError(f"Poster not found at {poster_file}")
                
        return redirect(url_for('display_poster', title=safe_title))
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('error.html', message=str(e)), 500

@app.route('/poster/<path:title>/<filename>')
def serve_poster(title, filename):
    try:
        # Décoder le titre URL-encodé
        decoded_title = unquote(title)
        directory_path = os.path.join(app.config['UPLOAD_FOLDER'], decoded_title)
        return send_from_directory(directory_path, filename)
    except FileNotFoundError:
        return "Poster not found", 404


@app.route('/display/<title>')
def display_poster(title):
    try:
        decoded_title = unquote(title)  # Déjà décodé par Flask, mais sécurité supplémentaire
        poster_files = [
            'poster_v1_5.png',
            'poster_v2_1.png'
        ]
        poster_paths = [url_for('serve_poster', title=title, filename=f) for f in poster_files]

        return render_template('display.html',
                            title=decoded_title,
                            criteria=FIXED_CRITERIA,
                            poster_paths=poster_paths)
    except Exception as e:
        return render_template('error.html', message=str(e)), 500
    
    
@app.route('/submit_rating/<title>', methods=['POST'])
def submit_rating(title):
    try:
        decoded_title = unquote(title)  # Décode le titre
        ratings = {criterion: int(request.form.get(criterion, 0)) 
                 for criterion in FIXED_CRITERIA}
        
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], decoded_title)
        os.makedirs(folder_path, exist_ok=True)  # Crée le dossier si inexistant
        
        csv_path = os.path.join(folder_path, 'ratings.csv')
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(','.join(FIXED_CRITERIA) + '\n')
                
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(','.join(map(str, ratings.values())) + '\n')
        
        return render_template('thanks.html')
    
    except Exception as e:
        return render_template('error.html', message=str(e)), 500

if __name__ == '__main__':
    print("Sous-application Flask en cours d'exécution sur http://127.0.0.1:5001/")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)