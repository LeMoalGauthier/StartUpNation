from controllers.movieApp import MovieApp
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
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
