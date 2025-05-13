# scripts/train_gpu.py
import tensorflow as tf
from keras import layers, models, callbacks
import os
import numpy as np
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def check_gpu_availability():
    """Diagnostic complet pour la détection et configuration du GPU"""
    print("==== Informations de diagnostic GPU ====")
    print(f"Version de TensorFlow: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Vérifier si TensorFlow peut voir le GPU
    print(f"GPU physiques disponibles selon TF: {tf.config.list_physical_devices('GPU')}")
    print(f"GPU est disponible pour TF: {tf.test.is_gpu_available()}")
    
    # Vérifier si CUDA est disponible
    print(f"Périphériques CUDA visibles: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Non défini')}")
    
    # Essayer de forcer l'utilisation du GPU si disponible
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Test de calcul sur GPU: {c}")
            print("Test de calcul GPU réussi!")
    except RuntimeError as e:
        print(f"Erreur lors du test du GPU: {e}")
        
    # Information sur le périphérique par défaut
    print(f"Périphérique par défaut: {tf.config.list_logical_devices()}")
    print("=======================================")

def build_improved_model():
    """Architecture de modèle améliorée avec plus de couches et une meilleure régularisation"""
    model = models.Sequential()
    
    # Premier bloc de convolution
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Deuxième bloc de convolution
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Troisième bloc de convolution
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Partie fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='softmax'))  # 7 émotions
    
    return model

def train_improved_model():
    """Entraîne un modèle amélioré avec plus d'optimisations"""
    print("Chargement et préparation des données...")
    X, y = load_and_preprocess_data("./data/train.csv")
    
    # Vérifier les dimensions
    print(f"Dimensions de X: {X.shape}")
    print(f"Dimensions de y: {y.shape}")
    
    # Séparer en train/val/test avec stratification
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"Train: {X_train.shape[0]} échantillons")
    print(f"Validation: {X_val.shape[0]} échantillons")
    print(f"Test: {X_test.shape[0]} échantillons")
    
    # Reshape pour les données d'entrée du modèle (ajouter dimension des canaux)
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Forcer l'utilisation du GPU si disponible
    try:
        with tf.device('/GPU:0'):
            # Construire le modèle amélioré
            model = build_improved_model()
            
            # Compiler avec un learning rate réduit
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Afficher le résumé du modèle
            model.summary()
            
            # Callbacks améliorés
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
            
            checkpoint = callbacks.ModelCheckpoint(
                filepath='./results/model/checkpoint_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            
            tensorboard_callback = callbacks.TensorBoard(
                log_dir="./results/logs",
                histogram_freq=1
            )
            
            # Créer le dossier de checkpoints s'il n'existe pas
            os.makedirs("./results/model", exist_ok=True)
            
            # Entraînement du modèle
            print("Démarrage de l'entraînement sur GPU...")
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=100,  # Plus d'époques avec early stopping
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_callback],
                verbose=1
            )
            
            # Évaluation sur l'ensemble de test
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
            print(f"Précision sur l'ensemble de test: {test_acc*100:.2f}%")
            
            # Sauvegarder le modèle final
            model.save("./results/model/final_emotion_model.keras")
            
            # Sauvegarder l'architecture
            with open("./results/model/final_emotion_model_arch.txt", "w") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            
            # Visualisation des courbes d'apprentissage
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Learning Curves - Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Learning Curves - Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("./results/model/learning_curves_1.png")
            
            # Essayer de créer une matrice de confusion si seaborn est disponible
            try:
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                # Prédire les classes pour l'ensemble de test
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
                
                # Créer la matrice de confusion
                cm = confusion_matrix(y_true_classes, y_pred_classes)
                
                # Afficher la matrice de confusion
                plt.figure(figsize=(10, 8))
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
                plt.xlabel('Prédictions')
                plt.ylabel('Réalité')
                plt.title('Matrice de Confusion')
                plt.savefig('./results/model/confusion_matrix.png')
            except ImportError:
                print("Seaborn n'est pas installé, la matrice de confusion ne sera pas générée")
    
    except RuntimeError as e:
        print(f"ERREUR: Impossible d'utiliser le GPU: {e}")
        print("Retour à l'utilisation du CPU...")
        
        # Continuer l'entraînement sur CPU
        model = build_improved_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entraînement du modèle sur CPU
        print("Démarrage de l'entraînement sur CPU...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
                callbacks.ModelCheckpoint('./results/model/checkpoint_best.keras', monitor='val_accuracy', save_best_only=True)
            ],
            verbose=1
        )
        
        # Évaluation sur l'ensemble de test
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
        print(f"Précision sur l'ensemble de test: {test_acc*100:.2f}%")
        
        # Sauvegarder le modèle final
        model.save("./results/model/final_emotion_model.keras")
    
    return model, history

if __name__ == "__main__":
    # Activer la journalisation TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Afficher tous les messages
    
    # Diagnostic GPU
    check_gpu_availability()
    
    # Définir des variables d'environnement pour la détection GPU
    # Décommenter ces lignes si nécessaire pour votre configuration
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Spécifier l'ID du GPU à utiliser
    
    # Configuration des GPU
    print("Configuration des GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponibles: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("Aucun GPU détecté selon tf.config.list_physical_devices")
        print("Tentative de forcer l'utilisation du GPU si disponible...")
    
    # Entraîner le modèle amélioré
    train_improved_model()