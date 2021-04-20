import tensorflow as tf
import sys

h5dir = sys.argv[1]
tflite_dir = sys.argv[2]

model = tf.keras.models.load_model(h5_dir, custom_objects={'tf':tf})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_dir, "wb").write(tflite_model)