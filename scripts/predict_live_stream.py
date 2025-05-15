import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import time
import os
from datetime import datetime

# Charger le modèle
model = load_model("results/model/final_emotion_model.keras")

# Labels d'émotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Chemin alternatif pour le classificateur Haar Cascade
# Option 1: Essayer plusieurs chemins possibles
haar_cascade_paths = [
    'haarcascade_frontalface_default.xml',  # Dans le répertoire courant
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None,
    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
    # Ajoutez d'autres chemins possibles selon votre système
]

# Trouver le premier chemin valide
face_cascade = None
for path in haar_cascade_paths:
    if path and os.path.exists(path):
        try:
            face_cascade = cv2.CascadeClassifier(path)
            if not face_cascade.empty():
                print(f"Classificateur chargé depuis: {path}")
                break
        except:
            continue

# Si aucun chemin n'a fonctionné, téléchargez le fichier
if face_cascade is None or face_cascade.empty():
    print("Impossible de trouver le classificateur Haar Cascade. Vous pouvez le télécharger manuellement depuis:")
    print("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
    print("et le placer dans le répertoire courant.")
    exit(1)

# Créer le dossier pour sauvegarder les images prétraitées s'il n'existe pas
preprocessing_folder = "results/preprocessing_test/"
os.makedirs(preprocessing_folder, exist_ok=True)
print(f"Les images prétraitées seront sauvegardées dans: {preprocessing_folder}")

# Initialiser la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam")
    exit(1)

# Variable pour suivre le temps de la dernière détection d'émotion
last_emotion_time = 0
emotion_cooldown = 0.5 # Temps minimum entre les détections d'émotions (en secondes)

# Variables pour la sauvegarde des images prétraitées
start_time = time.time()
saved_images_count = 0
recording_duration = 120  # Durée minimale d'enregistrement en secondes
required_images = recording_duration
save_image_interval = 0.80 # Intervalle entre les sauvegardes d'images
last_save_time = 0

print("Reading video stream ...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la frame de la webcam")
        break

    # Convertir en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Obtenir le temps actuel
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # # Afficher le temps écoulé
    # time_text = f"Temps écoulé: {elapsed_time:.1f}s"
    # cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Afficher le nombre d'images sauvegardées
    # images_text = f"Images sauvegardées: {saved_images_count}/{required_images}"
    # cv2.putText(frame, images_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    for (x, y, w, h) in faces:
        # Toujours dessiner le rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Vérifier si assez de temps s'est écoulé depuis la dernière détection
        if current_time - last_emotion_time >= emotion_cooldown:
            # Extraire et préparer la région du visage
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape (1, 48, 48, 1)

            # Prédire l'émotion
            preds = model.predict(face_input)
            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]
            prob = int(preds[0][emotion_idx] * 100)

            # Afficher l'émotion sur l'image
            label = f"{emotion}: {prob}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Afficher dans la console avec un horodatage
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} : {emotion}, {prob}%")
            
            # Mettre à jour le temps de la dernière détection
            last_emotion_time = current_time
            
            # Sauvegarder l'image prétraitée si l'intervalle est atteint et qu'on n'a pas encore assez d'images
            if saved_images_count < required_images and current_time - last_save_time >= save_image_interval:
                # Sauvegarder l'image prétraitée (normalisée)
                img_filename = f"{preprocessing_folder}face_{saved_images_count+1:02d}_{emotion}_{prob}.png"
                # Convertir l'image normalisée en format 8-bit pour l'enregistrement
                processed_img = (face_normalized * 255).astype(np.uint8)
                cv2.imwrite(img_filename, processed_img)
                saved_images_count += 1
                last_save_time = current_time
                print(f"Image sauvegardée: {img_filename}")
        else:
            # Pendant le délai, on affiche toujours la dernière émotion détectée mais sans refaire la prédiction
            if 'label' in locals():
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Afficher le temps restant avant la prochaine détection
    time_since_last = current_time - last_emotion_time
    if time_since_last < emotion_cooldown:
        cooldown_text = f"Prochaine analyse dans: {emotion_cooldown - time_since_last:.1f}s"
        cv2.putText(frame, cooldown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Affichage avec OpenCV
    cv2.imshow('Emotion Detection', frame)
    
    # Si nous avons sauvegardé suffisamment d'images et que le temps minimum est écoulé
    if saved_images_count >= required_images and elapsed_time >= recording_duration:
        print(f"\nTerminé! {saved_images_count} images ont été sauvegardées en {elapsed_time:.1f} secondes.")
        print(f"Les images sont disponibles dans le dossier: {preprocessing_folder}")
        # Optionnel: Ajouter un délai pour que l'utilisateur puisse voir le message de fin
        time.sleep(3)
        break
    
    # Si l'utilisateur appuie sur 'q', arrêter le programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()