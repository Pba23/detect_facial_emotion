"""
Script pour créer une attaque adversariale sur un CNN de classification d'émotions.
L'objectif est de modifier légèrement une image "Happy" pour que le modèle la classifie comme "Sad",
tout en conservant son apparence visuelle pour l'humain.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocess import load_and_preprocess_data
import os

# Configuration
TARGET_LABEL_FROM = 3  # Index de la classe "Happy" (0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral)
TARGET_LABEL_TO = 4    # Index de la classe "Sad"
MODEL_PATH = './results/model/final_emotion_model.keras'
CONFIDENCE_THRESHOLD = 0.9  # 90% de confiance minimum
EPSILON = 0.01  # Force de l'attaque (plus petit = changements plus subtils)
MAX_ITERATIONS = 500  # Nombre maximum d'itérations pour l'attaque

def load_data_and_model():
    """Charge les données et le modèle pré-entraîné"""
    print("Chargement des données...")
    X, y = load_and_preprocess_data("./data/train.csv")
    X = X.reshape(X.shape[0], 48, 48, 1)
    
    print("Chargement du modèle...")
    model = load_model(MODEL_PATH)
    model.summary()
    
    return X, y, model

def find_candidate_image(X, y, model):
    """Trouve une image 'Happy' avec une confiance >90%"""
    # Trouver les indices des images "Happy"
    happy_indices = np.where(np.argmax(y, axis=1) == TARGET_LABEL_FROM)[0]
    
    # Prédire les probabilités pour toutes les images "Happy"
    X_happy = X[happy_indices]
    predictions = model.predict(X_happy)
    
    # Trouver les images avec une confiance >90% pour "Happy"
    high_confidence_indices = np.where(predictions[:, TARGET_LABEL_FROM] > CONFIDENCE_THRESHOLD)[0]
    
    if len(high_confidence_indices) == 0:
        raise ValueError("Aucune image 'Happy' avec une confiance >90% n'a été trouvée!")
    
    # Prendre la première image avec haute confiance
    selected_idx = high_confidence_indices[0]
    original_idx = happy_indices[selected_idx]
    original_image = X[original_idx]
    
    print(f"Image trouvée! Indice: {original_idx}")
    print(f"Confiance pour 'Happy': {predictions[selected_idx, TARGET_LABEL_FROM]*100:.2f}%")
    
    return original_image, original_idx

def create_adversarial_example(model, original_image):
    """
    Crée un exemple adversarial en modifiant progressivement l'image
    pour maximiser la probabilité de la classe cible.
    """
    # Créer une copie de l'image et la convertir en tensor avec gradient
    image = tf.Variable(original_image[np.newaxis, ...], dtype=tf.float32)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Fonction pour calculer la perte (on veut maximiser la probabilité de "Sad")
    def loss_fn():
        logits = model(image, training=False)
        # On veut minimiser la probabilité de la classe source et maximiser celle de la classe cible
        target_loss = -tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot([TARGET_LABEL_TO], depth=7),
            logits=logits
        )
        return target_loss[0]
    
    # Calculer les prédictions initiales
    initial_pred = model.predict(image)
    initial_class = np.argmax(initial_pred[0])
    initial_conf = initial_pred[0, initial_class]
    print(f"Prédiction initiale: {emotion_labels[initial_class]} avec {initial_conf*100:.2f}% de confiance")
    
    # Attaque itérative
    prev_loss = float('-inf')
    i = 0
    
    while i < MAX_ITERATIONS:
        with tf.GradientTape() as tape:
            loss = loss_fn()
        
        # Calculer les gradients par rapport à l'image
        gradient = tape.gradient(loss, image)
        
        # Normaliser le gradient
        gradient = tf.sign(gradient)
        
        # Mettre à jour l'image en soustrayant le gradient (pour maximiser la perte)
        image.assign_add(EPSILON * gradient)
        
        # Clipper l'image pour qu'elle reste dans la plage [0, 1]
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        
        # Vérifier les prédictions actuelles
        if i % 50 == 0 or i == MAX_ITERATIONS - 1:
            pred = model.predict(image)
            current_class = np.argmax(pred[0])
            current_conf = pred[0, current_class]
            target_conf = pred[0, TARGET_LABEL_TO]
            
            print(f"Itération {i}:")
            print(f"  Classe prédite: {emotion_labels[current_class]} avec {current_conf*100:.2f}% de confiance")
            print(f"  Confiance pour 'Sad': {target_conf*100:.2f}%")
            
            # Si on a réussi à changer la classification vers "Sad" avec confiance suffisante
            if current_class == TARGET_LABEL_TO and target_conf > 0.5:
                print("Attaque réussie!")
                break
            
            # Si la perte ne s'améliore plus significativement
            current_loss = loss.numpy()
            if abs(current_loss - prev_loss) < 1e-6 and i > 100:
                print("Convergence atteinte (perte stable)")
                break
            prev_loss = current_loss
        
        i += 1
    
    return image.numpy()[0], model.predict(image)[0]

def visualize_results(original_image, adversarial_image, original_pred, adversarial_pred):
    """Visualise l'image originale et l'image adversariale avec leurs prédictions"""
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    plt.figure(figsize=(12, 5))
    
    # Image originale
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.reshape(48, 48), cmap='gray')
    plt.title(f"Original: {emotion_labels[np.argmax(original_pred)]}\n" + 
              f"Confiance: {np.max(original_pred)*100:.2f}%")
    plt.axis('off')
    
    # Image adversariale
    plt.subplot(1, 3, 2)
    plt.imshow(adversarial_image.reshape(48, 48), cmap='gray')
    plt.title(f"Adversarial: {emotion_labels[np.argmax(adversarial_pred)]}\n" + 
              f"Confiance: {np.max(adversarial_pred)*100:.2f}%")
    plt.axis('off')
    
    # Différence (amplifiée pour la visualisation)
    plt.subplot(1, 3, 3)
    diff = adversarial_image - original_image
    plt.imshow(diff.reshape(48, 48), cmap='RdBu_r')
    plt.title(f"Différence (amplifiée)\nMax: {np.max(np.abs(diff)):.4f}")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Créer le dossier si nécessaire
    os.makedirs('./results/adversarial', exist_ok=True)
    plt.savefig('./results/adversarial/adversarial_attack.png', dpi=200)
    plt.show()
    
    # Sauvegarder les images individuelles
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image.reshape(48, 48), cmap='gray')
    plt.axis('off')
    plt.savefig('./results/adversarial/original.png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(adversarial_image.reshape(48, 48), cmap='gray')
    plt.axis('off')
    plt.savefig('./results/adversarial/adversarial.png', bbox_inches='tight', pad_inches=0.1, dpi=200)

def compute_image_similarity(original, adversarial):
    """Calcule différentes métriques de similarité entre les images"""
    # MSE (Mean Squared Error)
    mse = np.mean((original - adversarial) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # L2 norm (distance euclidienne)
    l2_norm = np.linalg.norm(original - adversarial)
    
    # Pourcentage de changement moyen par pixel
    mean_percent_change = np.mean(np.abs(original - adversarial)) * 100
    
    # L∞ norm (changement maximum)
    l_inf_norm = np.max(np.abs(original - adversarial))
    
    print("\nMétriques de similarité:")
    print(f"- MSE: {mse:.8f}")
    print(f"- PSNR: {psnr:.2f} dB (Plus élevé = plus similaire)")
    print(f"- L2 norm: {l2_norm:.6f}")
    print(f"- Changement moyen par pixel: {mean_percent_change:.4f}%")
    print(f"- Changement maximum sur un pixel: {l_inf_norm:.6f}")
    
    return {
        'mse': mse,
        'psnr': psnr,
        'l2_norm': l2_norm,
        'mean_percent_change': mean_percent_change,
        'l_inf_norm': l_inf_norm
    }

def main():
    """Point d'entrée principal"""
    print("=== Création d'une attaque adversariale sur le CNN d'émotions ===")
    
    # Charger données et modèle
    X, y, model = load_data_and_model()
    
    # Trouver une image candidate (Happy avec confiance élevée)
    original_image, original_idx = find_candidate_image(X, y, model)
    
    # Calculer les prédictions initiales
    original_pred = model.predict(original_image[np.newaxis, ...])[0]
    
    # Créer l'exemple adversarial
    adversarial_image, adversarial_pred = create_adversarial_example(model, original_image)
    
    # Calculer des métriques de similarité
    similarity = compute_image_similarity(original_image, adversarial_image)
    
    # Vérifier si l'attaque a réussi
    if np.argmax(adversarial_pred) == TARGET_LABEL_TO:
        print("\n✅ ATTAQUE RÉUSSIE!")
        print(f"L'image est maintenant classifiée comme {['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][TARGET_LABEL_TO]}")
        print(f"avec une confiance de {adversarial_pred[TARGET_LABEL_TO]*100:.2f}%")
    else:
        print("\n❌ ATTAQUE ÉCHOUÉE")
        print(f"L'image est toujours classifiée comme {['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(adversarial_pred)]}")
    
    # Visualiser les résultats
    visualize_results(original_image, adversarial_image, original_pred, adversarial_pred)
    
    # Sauvegarder les images en binaire numpy pour réutilisation
    np.save('./results/adversarial/original_image.npy', original_image)
    np.save('./results/adversarial/adversarial_image.npy', adversarial_image)
    
    print("\nTerminé! Les résultats ont été sauvegardés dans le dossier './results/adversarial/'")

if __name__ == "__main__":
    main()