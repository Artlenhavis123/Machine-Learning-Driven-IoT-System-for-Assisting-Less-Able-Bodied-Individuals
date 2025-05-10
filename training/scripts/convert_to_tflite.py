import tensorflow as tf

model = tf.keras.models.load_model('models/2025-04-30_19-17-23/fall_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('fall_model.tflite', 'wb') as f:
    f.write(tflite_model)
