import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('models/2025-04-29_20-50-18/fall_model.h5')

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the model
with open('fall_model.tflite', 'wb') as f:
    f.write(tflite_model)
