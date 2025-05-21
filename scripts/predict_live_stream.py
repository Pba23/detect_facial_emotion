import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import time
import os
from datetime import datetime

# Fonction pour vérifier si le fichier vidéo existe
def check_video_file(file_path):
    if os.path.exists(file_path):
        return True
    print(f"Erreur: Le fichier {file_path} n'existe pas.")
    return False

try:
    # Charger le modèle
    model = load_model("results/model/final_emotion_model.keras")

    # Labels d'émotions
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Chemin alternatif pour le classificateur Haar Cascade
    haar_cascade_paths = [
        'haarcascade_frontalface_default.xml',  # Dans le répertoire courant
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None,
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
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
            except Exception as e:
                print(f"Erreur lors du chargement du classificateur depuis {path}: {e}")
                continue

    # Si aucun chemin n'a fonctionné, sortir
    if face_cascade is None or face_cascade.empty():
        print("Impossible de trouver le classificateur Haar Cascade. Vous pouvez le télécharger manuellement depuis:")
        print("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
        print("et le placer dans le répertoire courant.")
        exit(1)

    # Créer le dossier pour sauvegarder les images prétraitées s'il n'existe pas
    preprocessing_folder = "results/preprocessing_test/"
    os.makedirs(preprocessing_folder, exist_ok=True)
    print(f"Les images prétraitées seront sauvegardées dans: {preprocessing_folder}")

    # Configuration initiale de la source vidéo
    def setup_video_source(use_webcam=True, webcam_index=0, video_file=None):
        """Configure la source vidéo (webcam ou fichier)"""
        if use_webcam:
            print(f"Tentative d'ouverture de la webcam avec l'indice {webcam_index}...")
            cap = cv2.VideoCapture(webcam_index)
            if cap.isOpened():
                # Vérifier que la webcam fonctionne réellement en lisant une frame
                ret, test_frame = cap.read()
                if ret:
                    print(f"Webcam {webcam_index} initialisée avec succès.")
                    return cap, True
                else:
                    cap.release()
                    print(f"La webcam {webcam_index} ne renvoie pas d'image.")
            else:
                print(f"Impossible d'ouvrir la webcam avec l'indice {webcam_index}")
        
        # Si la webcam échoue ou n'est pas demandée, essayer le fichier vidéo
        if video_file and check_video_file(video_file):
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                print(f"Lecture du fichier vidéo: {video_file}")
                
                # Pour les fichiers vidéo, vérifier si c'est un WebM
                if video_file.lower().endswith('.webm'):
                    print("Format WebM détecté. Redimensionnement activé.")
                    # La vidéo sera redimensionnée lors de la lecture
                
                return cap, True
        
        return None, False

    # Essayer d'abord les webcams
    webcam_indices = [0]  # Essayer les indices de webcam courants
    cap = None
    video_source_ok = False
    
    # Essayer chaque webcam
    for idx in webcam_indices:
        cap, video_source_ok = setup_video_source(use_webcam=True, webcam_index=idx)
        if video_source_ok:
            break
    
    # Si aucune webcam ne fonctionne, essayer le fichier vidéo
    video_files_to_try = ["video.mp4", "video.webm", "video.avi", "video.mov"]
    if not video_source_ok:
        print("Aucune webcam disponible. Tentative d'ouverture des fichiers vidéo...")
        for video_file in video_files_to_try:
            cap, video_source_ok = setup_video_source(use_webcam=False, video_file=video_file)
            if video_source_ok:
                print(f"Fichier vidéo trouvé: {video_file}")
                break
    
    # Si aucune source vidéo ne fonctionne, quitter
    if not video_source_ok or cap is None:
        print("Aucune source vidéo disponible. Le programme va se terminer.")
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
    webcam_failed = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if not webcam_failed:
                # Si c'est la première erreur, essayer de basculer vers différents fichiers vidéo
                print("Problème avec la source vidéo. Tentative d'ouverture de fichiers vidéo alternatifs...")
                if cap is not None:
                    cap.release()
                
                video_found = False
                for video_file in ["video.mp4", "video.webm", "video.avi", "video.mov"]:
                    cap, video_source_ok = setup_video_source(use_webcam=False, video_file=video_file)
                    if video_source_ok:
                        webcam_failed = True
                        video_found = True
                        print(f"Source vidéo alternative trouvée: {video_file}")
                        break
                
                if video_found:
                    continue  # Continuer avec le fichier vidéo
                else:
                    print("Impossible de trouver une source vidéo alternative.")
            else:
                print("Fin de la vidéo ou problème de lecture du flux")
            break

        # Vérifier si nous utilisons un fichier WebM (souvent trop grand)
        # Pour vérifier si une vidéo est en cours de lecture (pas une webcam)
        is_file_video = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        
        # Redimensionner l'image si c'est une vidéo (particulièrement pour les WebM)
        if is_file_video:
            # Redimensionner pour éviter le zoom excessif
            # Obtenir la taille actuelle de l'image
            height, width = frame.shape[:2]
            
            # Si la taille est très grande (typique pour certains fichiers WebM), réduire
            max_width = 1000  # Largeur maximale souhaitée
            if width > max_width:
                # Calculer le ratio pour maintenir les proportions
                ratio = max_width / width
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Image redimensionnée de {width}x{height} à {new_width}x{new_height}", end='\r')

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Obtenir le temps actuel
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        for (x, y, w, h) in faces:
            # Dessiner le rectangle autour du visage
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

    # Libérer les ressources
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
except Exception as e:
    print(f"Une erreur est survenue: {e}")
    import traceback
    traceback.print_exc()
    # Assurer que les ressources sont libérées même en cas d'erreur
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
