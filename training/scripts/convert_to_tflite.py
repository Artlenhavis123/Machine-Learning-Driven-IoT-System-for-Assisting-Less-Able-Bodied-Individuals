import tensorflow as tf

# Load your Keras .h5 model
model = tf.keras.models.load_model('models/2025-04-27_23-35-44/fall_model.h5')

# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Set optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the .tflite model
with open('fall_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model successfully converted to fall_model.tflite")