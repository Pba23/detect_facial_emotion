import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

# Charger et prétraiter les données
X_test, y_test = load_and_preprocess_data("data/test_with_emotions.csv")

# Charger le modèle
model = load_model("results/model/final_emotion_model.keras")

# Prédictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Corriger la forme de y_test si one-hot
y_test_labels = np.argmax(y_test, axis=1)

# Évaluer
acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Accuracy on test set: {acc * 100:.2f}%")
