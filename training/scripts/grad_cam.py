import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("models/fall_model.h5")
_ = model(tf.zeros((1, 224, 224, 3)))

# Grad-CAM function
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Apply heatmap to image
def superimpose_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# Load image
def load_preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Main function to run Grad-CAM
def run_gradcam_on_image(img_path, output_path, conv_layer='conv2d_1'):
    img_array = load_preprocess_image(img_path)
    heatmap = get_gradcam_heatmap(model, img_array, conv_layer)
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    cam_image = superimpose_heatmap(heatmap, original)
    cv2.imwrite(output_path, cam_image)
    print(f"✅ Saved Grad-CAM image to {output_path}")

# Example usage:
if __name__ == "__main__":
    test_image_path = "datasets/split_frames/test/Fall/fall-15_0045.jpg"
    output_path = "models/gradcam_output.jpg"
    run_gradcam_on_image(test_image_path, output_path)