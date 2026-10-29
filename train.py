"""
Train a MobileNetV2 transfer-learning classifier.
Expected folders:
data/train/<class_name>/
data/val/<class_name>/
data/test/<class_name>/
Run from the project root: python train.py
"""
from pathlib import Path

DATASET_DIR = Path(
    r"C:\Users\ARJUN MT\Downloads\archive (2)\Plantvillage"
)

import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# Use the inner PlantVillage folder containing the class folders
DATASET_DIR = Path(
    r"C:\Users\ARJUN MT\Downloads\archive (2)\Plantvillage\PlantVillage"
)

os.makedirs("models", exist_ok=True)

# Load all classes and reserve 20% for validation and testing
full_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
    validation_split=0.30,
    subset="training",
)

temp_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
    validation_split=0.30,
    subset="validation",
)

class_names = full_ds.class_names

# Split the reserved 30% into validation and test datasets
temp_size = tf.data.experimental.cardinality(temp_ds).numpy()
val_batches = temp_size // 2

val_ds = temp_ds.take(val_batches)
test_ds = temp_ds.skip(val_batches)
train_ds = full_ds

with open("models/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Number of classes:", len(class_names))
print("Classes:", class_names)



with open("models/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("models/plant_disease.keras")

plt.figure()
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("models/accuracy.png")
plt.close()

test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probabilities = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
print(classification_report(
    y_true, y_pred, labels=list(range(len(class_names))),
    target_names=class_names, zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()
