import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Settings
base_dir = "datasets/split_frames"
model_path = "models/2025-04-22_01-24-20/fall_model.h5"
threshold = 0.5  # for sigmoid; ignore if using softmax
false_positive_dir = "misclassified/false_positives"
false_negative_dir = "misclassified/false_negatives"

# Prepare output folders
os.makedirs(false_positive_dir, exist_ok=True)
os.makedirs(false_negative_dir, exist_ok=True)

# Load model
model = load_model(model_path)

# Load test data
img_height, img_width = 224, 224
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",  # assumes softmax output
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes
filenames = test_generator.filenames

# Extract false positives (pred: Fall, true: No_Fall)
for idx, (pred, true) in enumerate(zip(pred_classes, true_classes)):
    if pred == 1 and true == 0:
        src = os.path.join(test_generator.directory, filenames[idx])
        dst = os.path.join(false_positive_dir, os.path.basename(filenames[idx]))
        shutil.copy2(src, dst)

# Extract false negatives (pred: No_Fall, true: Fall)
for idx, (pred, true) in enumerate(zip(pred_classes, true_classes)):
    if pred == 0 and true == 1:
        src = os.path.join(test_generator.directory, filenames[idx])
        dst = os.path.join(false_negative_dir, os.path.basename(filenames[idx]))
        shutil.copy2(src, dst)

print(f"✅ Extracted false positives: {len(os.listdir(false_positive_dir))}")
print(f"✅ Extracted false negatives: {len(os.listdir(false_negative_dir))}")