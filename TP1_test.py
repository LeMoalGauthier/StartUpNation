import numpy as np

# Définition des actions possibles
actions = ["Pierre", "Papier", "Ciseaux"]
n_actions = len(actions)

# Matrice de gains pour le joueur 1 (ligne = action joueur 1, colonne = action joueur 2)
matrice_gains = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

profils_joueur1 = [
    [1/3, 1/3, 1/3],
    [0, 2/3, 1/3],
    [1/6, 1/6, 2/3],
    [0, 1, 0]
]

profils_joueur2 = [
    [1/3, 1/3, 1/3],
    [1, 0, 0],
    [5/6, 1/6, 0],
    [2/3, 1/6, 1/6]
]

def obtenir_profil(joueur, iteration):
    # Changer de profil tous les 25 tours
    index_profil = (iteration // 25) % 4
    return profils_joueur1[index_profil] if joueur == 1 else profils_joueur2[index_profil]


def fictitious_play(matrice_gains, n_iterations=100):
    freq_joueur1 = np.zeros(n_actions)
    freq_joueur2 = np.zeros(n_actions)
    
    historique_joueur1 = []
    historique_joueur2 = []
    strategie_joueur1 = obtenir_profil(1, 1)
    strategie_joueur2 = obtenir_profil(2, 1)
    
    for t in range(n_iterations):
        gain_espere_joueur1 = matrice_gains @ strategie_joueur2
        gain_espere_joueur2 = -matrice_gains.T @ strategie_joueur1
        
        action_joueur1 = np.argmax(gain_espere_joueur1)
        action_joueur2 = np.argmax(gain_espere_joueur2)
        
        freq_joueur1[action_joueur1] += 1
        freq_joueur2[action_joueur2] += 1
        
        historique_joueur1.append(action_joueur1)
        historique_joueur2.append(action_joueur2)
        
        print(f"Iteration {t+1}: Joueur 1 (profil {strategie_joueur1}) choisit {actions[action_joueur1]}, "
              f"Joueur 2 (profil {strategie_joueur2}) choisit {actions[action_joueur2]}")
    
        strategie_joueur1 = freq_joueur1 / t
        strategie_joueur2 = freq_joueur2 / t
    
    return strategie_joueur1, strategie_joueur2

n_iterations = 100
strategie_finale_joueur1, strategie_finale_joueur2 = fictitious_play(matrice_gains, n_iterations)

# Affichage des résultats finaux
print("\nStratégie mixte finale pour le Joueur 1 :", strategie_finale_joueur1)
print("Stratégie mixte finale pour le Joueur 2 :", strategie_finale_joueur2)
