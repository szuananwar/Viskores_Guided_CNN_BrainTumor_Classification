import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical

os.makedirs("results", exist_ok=True)

IMAGE_DIR = "data/archive/Training"
FEATURE_FILE = "data/features/enhanced_features.csv"

IMG_SIZE = (128, 128)
EPOCHS = 10
BATCH_SIZE = 32

df = pd.read_csv(FEATURE_FILE)

feature_cols = [
    "isosurface_points",
    "isosurface_cells",
    "mean_intensity",
    "median_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "q25_intensity",
    "q75_intensity",
    "intensity_range",
    "iqr_intensity",
    "voxel_count",
    "high_intensity_voxels",
    "voxel_ratio",
    "high_intensity_ratio",
    "point_cell_ratio",
    "cell_density",
    "point_density"
]

images = []
labels = []

class_order = ["glioma", "meningioma", "notumor", "pituitary"]

for class_name in class_order:
    class_folder = os.path.join(IMAGE_DIR, class_name)

    if not os.path.exists(class_folder) and class_name == "notumor":
        class_folder = os.path.join(IMAGE_DIR, "no_tumor")

    files = sorted([
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for filename in files[:1400]:
        img_path = os.path.join(class_folder, filename)

        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img = np.array(img) / 255.0

        images.append(img)
        labels.append(class_name)

X_img = np.array(images, dtype=np.float32)
X_feat = df[feature_cols].values
y = np.array(labels)

print("Image shape:", X_img.shape)
print("Feature shape:", X_feat.shape)
print("Labels shape:", y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

scaler = StandardScaler()
X_feat_scaled = scaler.fit_transform(X_feat)

X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_img,
    X_feat_scaled,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Image CNN branch
image_input = Input(shape=(128, 128, 3), name="mri_image_input")

x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Viskores feature branch
feature_input = Input(shape=(len(feature_cols),), name="viskores_feature_input")

f = layers.Dense(64, activation="relu")(feature_input)
f = layers.Dense(32, activation="relu")(f)

# Fusion
combined = layers.concatenate([x, f])

z = layers.Dense(128, activation="relu")(combined)
z = layers.Dropout(0.4)(z)
z = layers.Dense(64, activation="relu")(z)

output = layers.Dense(4, activation="softmax")(z)

model = models.Model(
    inputs=[image_input, feature_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    [X_img_train, X_feat_train],
    y_train,
    validation_data=([X_img_test, X_feat_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

pred_probs = model.predict([X_img_test, X_feat_test])
preds = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test, axis=1)

class_names = encoder.classes_

report = classification_report(
    true_labels,
    preds,
    target_names=class_names
)

print(report)

with open("results/feature_fusion_cnn_classification_report.txt", "w") as f:
    f.write(report)

accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds, average="weighted")
recall = recall_score(true_labels, preds, average="weighted")
f1 = f1_score(true_labels, preds, average="weighted")

metrics = pd.DataFrame([{
    "model": "Pipeline 2 - Viskores Feature Fusion CNN",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

metrics.to_csv("results/feature_fusion_cnn_metrics.csv", index=False)

cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - Viskores Feature Fusion CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/feature_fusion_cnn_confusion_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Feature Fusion CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/feature_fusion_cnn_accuracy_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Feature Fusion CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/feature_fusion_cnn_loss_curve.png", dpi=300)
plt.close()

model.save("results/feature_fusion_cnn.keras")

print("Feature fusion CNN results saved.")
print("Accuracy:", accuracy)
