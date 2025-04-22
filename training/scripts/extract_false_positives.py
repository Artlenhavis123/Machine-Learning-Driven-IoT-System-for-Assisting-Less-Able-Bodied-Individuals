import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Settings
base_dir = "datasets/split_frames"
model_path = "models/2025-04-22_00-44-41/fall_model.h5"  # Update this if using a different model
output_dir = "misclassified/false_positives"
threshold = 0.5  # For sigmoid model; adjust if using softmax

# Prepare output folder
os.makedirs(output_dir, exist_ok=True)

# Load model
model = load_model(model_path)

# Prepare test data
img_height, img_width = 224, 224
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
pred_classes = (preds.flatten() > threshold).astype(int)
true_classes = test_generator.classes
filenames = test_generator.filenames

# Get label mapping
label_map = {v: k for k, v in test_generator.class_indices.items()}

# Copy false positives
for idx, (pred, true) in enumerate(zip(pred_classes, true_classes)):
    if pred == 1 and true == 0:
        src = os.path.join(test_generator.directory, filenames[idx])
        dest = os.path.join(output_dir, os.path.basename(filenames[idx]))
        shutil.copy2(src, dest)

print(f"✅ Extracted {len(os.listdir(output_dir))} false positive images to {output_dir}")