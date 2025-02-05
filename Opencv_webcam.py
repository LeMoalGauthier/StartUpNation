import cv2
import mediapipe as mp
import csv
import os
import time

# Initialiser MediaPipe Hands et FaceMesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils  # Outil pour dessiner les points clés

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Liste des indices des points importants pour la LSF (mains et visage)
MAIN_POINTS = [0, 4, 8, 12, 16, 20]  # Pouce, index, majeur, annulaire, auriculaire
FACE_POINTS = [33, 263, 61, 291, 199]  # Yeux, lèvres

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam")
    exit()

# Fichier CSV pour stocker les données
csv_file = "coordonnees_mains_visage.csv"

# Définition des en-têtes du CSV (21 points pour la main + 5 points essentiels pour le visage)
header = ["frame"] + [f"x_hand{i},y_hand{i},z_hand{i}" for i in range(21)] + \
         [f"x_face{i},y_face{i},z_face{i}" for i in range(len(FACE_POINTS))]

# Vérifier si le fichier existe déjà, sinon créer l'entête
file_exists = os.path.exists(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)  # Ajouter l'entête si fichier inexistant

frame_count = 0  # Compteur d'images
sampling_rate = 1  # Capture une frame toutes les images

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image")
        break  

    # Convertir en RGB (MediaPipe attend du RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains et du visage
    hands_result = hands.process(rgb_frame)
    face_result = face_mesh.process(rgb_frame)

    # Ajuster la fréquence d’échantillonnage
    if frame_count % sampling_rate == 0:
        row = [frame_count] + [""] * (21 * 3) + [""] * (len(FACE_POINTS) * 3)

        # Détection des mains
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                for i in MAIN_POINTS:
                    landmark = hand_landmarks.landmark[i]
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Vert pour les mains

                    row[1 + i * 3] = landmark.x
                    row[2 + i * 3] = landmark.y
                    row[3 + i * 3] = landmark.z  # Profondeur

        # Détection du visage et des points essentiels
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                for i, idx in enumerate(FACE_POINTS):
                    landmark = face_landmarks.landmark[idx]
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # Bleu pour les points du visage

                    row[64 + i * 3] = landmark.x  # Décalage après les points des mains
                    row[65 + i * 3] = landmark.y
                    row[66 + i * 3] = landmark.z

        # Enregistrement des données dans le CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    frame_count += 1  # Incrémenter le compteur

    # Afficher l'image avec les points essentiels
    cv2.imshow("Webcam - Détection des Mains & Points Essentiels du Visage", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()