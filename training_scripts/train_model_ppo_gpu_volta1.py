import training_ppo as training
import ray
import ray.rllib.agents.ppo as ppo
import json, os, shutil, sys
import gym
import pprint
import time
import shelve
from tensorflow import keras
from ray import tune
import tensorflow as tf

#shutil.rmtree('~/ray_results', ignore_errors = True, onerror = False)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices, 'GPU')
print("Available Physical GPUs: {}".format(physical_devices))
gpu_options = sys.argv[1]
ray.shutdown()
model = sys.argv[2]
config = ppo.DEFAULT_CONFIG.copy()
num_workers = int(sys.argv[3])
if(gpu_options == 'gpu0'):
    # Set only GPU 0 as visible
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    ray.init(num_gpus=1)
    num_gpus = 1
    config['num_workers'] = num_workers
    config['num_gpus'] = 0.0001
    config['num_gpus_per_worker'] = (num_gpus-config['num_gpus'])/num_workers

if(gpu_options == 'gpu1'):
    # Set only GPU 0 as visible
    tf.config.set_visible_devices(physical_devices[1], 'GPU')
    ray.init(num_gpus=1)
    num_gpus=1
    config['num_workers'] = num_workers
    config['num_gpus'] = 0.0001
    config['num_gpus_per_worker'] = (num_gpus-config['num_gpus'])/num_workers
   
if(gpu_options == 'both'):
    ray.init(num_gpus=2)
    num_gpus=2
    config['num_workers'] = num_workers
    config['num_gpus'] = 0.0001
    config['num_gpus_per_worker'] = (num_gpus-config['num_gpus'])/num_workers

#logical_devices = tf.config.list_logical_devices('GPU')
#print("Available logical GPUs: {}".format(logical_devices))

save_name = sys.argv[4]

if model == 'model1':
    save_file = './training_results/ppo/model1/' + save_name
    checkpoint_root='./checkpoints/ppo/' + save_name
elif model == 'model2':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [16, 16], 8],[32, [4, 4], 2],[256, [11, 11], 1]]
    save_file = './training_results/ppo/model2/' + save_name
    checkpoint_root='./checkpoints/ppo/' + save_name
elif model == 'model3':
    config['model']['dim'] = 252
    config['model']['conv_filters'] = [[16, [8, 8], 4],[16, [8, 8], 4], [32, [4, 4], 2], [256, [8, 8], 1]]
    save_file = './training_results/ppo/model3/' + save_name
    checkpoint_root='./checkpoints/ppo/' + save_name
elif model == 'model4':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2],[32, [4, 4], 2], [256, [11, 11], 1]]
    save_file = './training_results/ppo/model4/'+ save_name
    checkpoint_root='./checkpoints/ppo/' + save_name
elif model == 'model5':
    config['model']['dim'] = 252
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2], [32, [4, 4], 2], [256, [16, 16], 1]]
    save_file = './training_results/ppo/model5/' + save_name
    checkpoint_root='./checkpoints/ppo/' + save_name
elif model == 'model6':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2],[256, [21, 21], 1]]
    save_file = './training_results/ppo/model6/'+ save_name
    checkpoint_root='./checkpoints/ppo/' + save_name

agent = ppo.PPOTrainer(config, env='Pong-v0')
policy=agent.get_policy()
print(policy.model.model_config)
print(policy.model.base_model.summary())

print("Configuración del agente:\n\n" + str(config))
print("\nConfiguración del modelo del agente:\n\n" + str(config["model"]))

t0 = time.time()
n_iter = int(sys.argv[5])
training.full_train(checkpoint_root, agent, n_iter, save_file)
t1 = time.time()-t0
print("Total time for the " + str(n_iter) + " training iterations: " + str(t1))