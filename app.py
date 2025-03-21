import subprocess
from controllers.route import app

if __name__ == '__main__':
    # Lancer la sous-application en arrière-plan
    subprocess.Popen(['python', 'sous_app/app.py'])
    
    # Lancer l'application principale
    app.run(debug=True)