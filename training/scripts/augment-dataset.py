import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# CONFIG
dataset_dir = 'training/datasets/frames'
classes_to_augment = ['Fall', 'Pre_Fall', 'Idle', 'Lying', 'Sitting'] 
target_count = 2000  # how many total images you want per class after augmenting
output_suffix = '_aug'

# Set up augmentations
augmenter = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)

for class_name in classes_to_augment:
    class_path = os.path.join(dataset_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg'))]
    current_count = len(images)
    print(f"Augmenting '{class_name}' from {current_count} to {target_count} images")

    i = 0
    while current_count + i < target_count:
        img_path = os.path.join(class_path, images[i % current_count])
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Generate one image
        for batch in augmenter.flow(img_array, batch_size=1):
            new_img = array_to_img(batch[0])
            new_filename = f"{os.path.splitext(images[i % current_count])[0]}{output_suffix}_{i}.jpg"
            new_img.save(os.path.join(class_path, new_filename))
            i += 1
            break
