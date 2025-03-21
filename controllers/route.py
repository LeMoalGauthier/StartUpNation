from controllers.movieApp import MovieApp
from flask import Flask, render_template, request, jsonify, redirect
import os
import subprocess
import requests
import sys
import time

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
movie_app = MovieApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_movie_info')
def get_movie_info_route():
    movie_title = request.args.get('title')
    movies_info = movie_app.get_movies_info(movie_title)
    return jsonify(movies_info)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = movie_app.get_movie_details(movie_id)
    if movie:
        return render_template('movie_details.html', movie=movie)
    return "Film non trouvé", 404

def start_subapp():
    try:
        requests.get("http://127.0.0.1:5001/")
    except requests.ConnectionError:
        subapp_path = os.path.abspath("sous_app/app.py")
        subprocess.Popen([sys.executable, subapp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.route('/launch_subapp')
def launch_subapp():
    start_subapp()
    time.sleep(2)
    return redirect("http://127.0.0.1:5001/")