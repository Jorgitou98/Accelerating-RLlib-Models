import ray
import ray.cloudpickle as cloudpickle
import ray.rllib.agents.ppo as ppo
import sys
import os
import tensorflow as tf
from tensorflow import keras


ray.shutdown()
ray.init()
checkpoint_dir=sys.argv[1]
export_name = sys.argv[2]

config = ppo.DEFAULT_CONFIG.copy()
print(config)

config_dir = os.path.dirname(checkpoint_dir)
config_path = os.path.join(config_dir, "params.pkl")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "../params.pkl")
with open(config_path, "rb") as f:
    config = cloudpickle.load(f)
print(config)
config['num_gpus']=0
config['num_gpus_per_worker'] = 0
config['explore'] = False

agent = ppo.PPOTrainer(config, env='Pong-v0')
agent.restore(checkpoint_dir)
print(agent.get_policy().model.base_model.summary())

with agent.get_policy().get_session().graph.as_default():
    export_model = agent.get_policy().model.base_model.save(export_name + '.h5')

ray.shutdown()