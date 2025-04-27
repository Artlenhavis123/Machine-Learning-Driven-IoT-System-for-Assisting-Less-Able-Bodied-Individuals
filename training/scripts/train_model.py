import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import seaborn as sns
import datetime
import sys

# Setup output paths
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("models", timestamp)
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "results.txt")

# Log to file and console simultaneously
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file_path)

# Dataset parameters
base_dir = "datasets/split_frames"
img_height, img_width = 224, 224
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)
val_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)
test_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Class distribution
print("🔍 Class Distribution in Training Data:")
print(Counter(train_data.classes))

# Class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("🧮 Computed Class Weights:", class_weight_dict)

# CNN architecture
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

plot_model(model, to_file=f"models/cnn_architecture_{timestamp}.png", show_shapes=True, dpi=120)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Training plots
def plot_training_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Over Epochs')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

plot_training_curves(history)

# Evaluation
test_loss, test_acc = model.evaluate(test_data)
print(f"🧪 Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Predictions
y_pred = model.predict(test_data)
y_true = test_data.classes
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(test_data.class_indices.keys())

# Report
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("\n📄 Classification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\n🔢 Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Save model
model.save(os.path.join(output_dir, "fall_model.h5"))
print(f"✅ Model training complete. All outputs saved to {output_dir}")