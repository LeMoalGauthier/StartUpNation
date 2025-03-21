import pyttsx3

def lire_message(message):
    # Initialisation du moteur de synthèse vocale
    moteur = pyttsx3.init()

    # Configuration de la voix (optionnel)
    voix = moteur.getProperty('voices')
    moteur.setProperty('voice', voix[0].id)  # Change à voix[1] pour une autre voix
    moteur.setProperty('rate', 150)  # Vitesse de lecture (plus bas = plus lent)

    # Lire le message
    moteur.say(message)
    moteur.runAndWait()

# Exemple d'utilisation
message = input("Entrez le message à lire : ")
lire_message(message)