# Projet de Détection d'Émotions Faciales

Ce projet implémente un système de détection d'émotions à partir d'expressions faciales en utilisant des réseaux de neurones convolutifs (CNN). Le système peut reconnaître 7 émotions différentes : Joie, Tristesse, Colère, Surprise, Peur, Dégoût et Neutre.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Approche technique](#approche-technique)
6. [Résultats](#résultats)
7. [Fonctionnalités avancées](#fonctionnalités-avancées)
8. [Améliorations possibles](#améliorations-possibles)

## Vue d'ensemble

Ce projet a été développé dans le cadre d'une étude sur la reconnaissance d'émotions faciales en temps réel. Il combine des techniques de vision par ordinateur pour la détection de visages et des réseaux de neurones profonds pour la classification des émotions. Le système peut:

- Détecter un visage dans un flux vidéo
- Prétraiter l'image du visage
- Classifier l'émotion en temps réel
- Générer des exemples adversariaux (fonctionnalité avancée)

## Structure du projet

```
project/
├── data/
│   ├── test.csv
│   ├── train.csv
│   └── test_with_emotions.csv
├── requirements.txt
├── README.md
├── results/
│   ├── model/
│   │   ├── learning_curves.png
│   │   ├── final_emotion_model_arch.txt
│   │   ├── final_emotion_model.keras
│   │   └── confusion_matrix.png
│   ├── preprocessing_test/
│   │   ├── image0.png
│   │   ├── image1.png
│   │   └── input_video.mp4
│   └── adversarial/
│       ├── adversarial_attack.png
│       ├── original.png
│       └── adversarial.png
└── scripts/
    ├── validation_loss_accuracy.py
    ├── predict_live_stream.py
    ├── predict.py
    ├── preprocess.py
    ├── train.py
    └── hack_cnn.py
```

## Installation

Pour installer et configurer le projet:

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/emotion-detection.git
cd emotion-detection

# Installer les dépendances
pip install -r requirements.txt
```

### Prérequis

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- numpy
- matplotlib
- scikit-learn
- seaborn (optionnel, pour les visualisations)

## Utilisation

### 1. Telechargement du dataset 
Assure toi de donner les autorisation d'execution au fichier download_data.sh puis:
```bash
./download_data.sh
```

### 2. Entraînement du modèle

Pour entraîner le modèle CNN de détection d'émotions:

```bash
python scripts/train.py
```

Le modèle entraîné sera sauvegardé dans `results/model/final_emotion_model.keras`.

### 3. Évaluation du modèle

Pour évaluer les performances du modèle sur l'ensemble de test:

```bash
python scripts/predict.py
```

Sortie attendue:
```
Accuracy on test set: 62% (une valeur superieur à 60 est requise)
```

### 4. Détection d'émotions en temps réel

Pour lancer la détection d'émotions à partir du flux vidéo de la webcam:

```bash
python scripts/predict_live_stream.py
```

Sortie attendue:
```
Reading video stream ...

Preprocessing ...
11:11:11s : Happy , 73%

Preprocessing ...
11:11:12s : Happy , 93%

...
```

### 5. Génération d'exemples adversariaux (fonctionnalité avancée)

Pour générer un exemple adversarial qui trompe le modèle:

```bash
python scripts/hack_cnn.py
```

Les résultats seront sauvegardés dans `results/adversarial/`.

## Approche technique

### Prétraitement des données

1. Les images de visages sont redimensionnées à 48x48 pixels
2. Conversion en niveaux de gris
3. Normalisation des valeurs de pixels entre 0 et 1
4. Utilisation de OpenCV pour la détection de visages dans le flux vidéo
5. Augmentation de données pendant l'entraînement pour améliorer la généralisation

### Architecture du modèle CNN

Notre architecture CNN comporte:

- 3 blocs de convolution (avec BatchNormalization et MaxPooling)
- Couches de Dropout pour réduire le surapprentissage
- Couches Dense pour la classification finale
- Fonction d'activation softmax en sortie pour la prédiction des 7 classes d'émotions

L'architecture complète est détaillée dans `results/model/final_emotion_model_arch.txt`.

### Entraînement

- Optimiseur: Adam avec learning rate adaptatif
- Fonction de perte: Categorical Crossentropy
- Early stopping pour éviter le surapprentissage
- Monitoring avec TensorBoard
- Support du GPU pour accélérer l'entraînement

## Résultats

![Matrice de confusion disponible](results/model/confusion_matrix.png)
![Courbes d'apprentissage](results/model/learning_curves.png)
![TensorBoard Screenshot](results/model/tensorboard_screenshot.png)

### Distribution des prédictions par classe

Le modèle présente de bonnes performances sur les émotions distinctives comme la joie et la surprise, mais rencontre plus de difficultés pour distinguer la peur et la tristesse.

## Fonctionnalités avancées

### Attaque adversariale

Le script `hack_cnn.py` implémente une technique d'attaque adversariale qui démontre une vulnérabilité intéressante des réseaux de neurones. Il permet de:

1. Sélectionner une image classifiée comme "Happy" avec une confiance élevée (>90%)
2. Modifier subtilement les pixels de l'image pour que le modèle la classifie comme "Sad"
3. Les modifications sont imperceptibles à l'œil humain mais suffisantes pour tromper le modèle

Cette fonctionnalité illustre l'importance de la robustesse des modèles de deep learning face aux attaques adversariales.
![Attaque de l'adversaire](results/adversarial/adversarial_attack.png)

## Améliorations possibles

- Utilisation de modèles pré-entraînés (transfer learning) pour améliorer la précision
- Implémentation de techniques d'attention pour mieux se concentrer sur les régions faciales expressives
- Déploiement sur des appareils mobiles (TensorFlow Lite)
- Intégration avec d'autres modalités (audio) pour une reconnaissance multimodale des émotions
- Amélioration de la robustesse contre les attaques adversariales
