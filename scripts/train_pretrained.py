# train_pretrained.py
import pickle
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split

X, y = load_and_preprocess_data("./data/train.csv", to_rgb=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

base_model = MobileNetV2(include_top=False, input_shape=(48, 48, 3), weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# Ne pas entraîner les couches pré-entraînées au début
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)
# Sauvegarde au format pickle
with open("./results/model/pre_trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Sauvegarde architecture
with open("./results/model/pre_trained_model_architecture.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

model.save("results/model/pre_trained_model.keras")
