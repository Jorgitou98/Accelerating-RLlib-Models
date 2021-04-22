import sys
h5_dir = sys.argv[1]

images = []
with open('prueba.npy', 'rb') as f:
    for _ in range(100):
        images.append(np.load(f))

import tensorflow as tf 
def representative_data_gen():
    for data in tf.data.Dataset.from_tensor_slices((image)).batch(1).take(100):
        yield[tf.dtypes.cast(data, tf.float32)]

model = tf.keras.models.load_model(h5_dir, custom_objects={'tf':tf})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
onverter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()