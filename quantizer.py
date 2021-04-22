
import ray
import ray.rllib.agents.ppo as ppo
import numpy as np

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config['create_env_on_driver'] = True
config['model']['dim'] = 252
config['model']['conv_filters'] = [[16, [8, 8], 4],[16, [8, 8], 4], [32, [4, 4], 2], [256, [8, 8], 1]]
agent = ppo.PPOTrainer(env='Pong-v0', config = config)

env = agent.workers.local_worker().env
obs = env.reset()
images = []
for _ in range(100):
    images.append(obs)
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)

h5_dir = sys.argv[1]
ray.shutdown()

import tensorflow as tf 
model = tf.keras.models.load_model(h5_dir, custom_objects={'tf':tf})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()
#open(tflite_dir, "wb").write(tflite_model)

def representative_dataset():
    for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
        yield[tf.dtypes.cast(data, tf.float32)]


