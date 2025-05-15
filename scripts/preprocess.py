# scripts/preprocess.py
import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path, to_rgb=False):
    import pandas as pd
    import numpy as np
    from keras.utils import to_categorical

    # Charger les données
    data = pd.read_csv(csv_path)

    # Extraire les pixels et les émotions
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].values

    # Convertir les pixels en tableau
    X = np.array([np.fromstring(pixel_string, sep=' ') for pixel_string in pixels])
    X = X / 255.0

    # Reshape en (48, 48, 1)
    X = X.reshape(-1, 48, 48, 1)

    if to_rgb:
        # Convertir en 3 canaux RGB (répéter 3 fois le canal gris)
        X = np.repeat(X, 3, axis=-1)  # (n_samples, 48, 48, 3)

    # One-hot encode les labels
    y = to_categorical(emotions, num_classes=7)

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data("../data/train.csv")
    print("Shape des images:", X.shape)
    print("Shape des labels:", y.shape)

