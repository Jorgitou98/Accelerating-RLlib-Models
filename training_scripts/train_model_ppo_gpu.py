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

shutil.rmtree('~/ray_results', ignore_errors = True, onerror = False)
ray.shutdown()
ray.init()
model = sys.argv[1]
config = ppo.DEFAULT_CONFIG.copy()
num_workers = int(sys.argv[2])
config['num_workers'] = num_workers
config['num_gpus'] = 0.0001
config['num_gpus_per_worker'] = 0.9999/num_workers

if model == 'model1':
    save_file = './training_results/ppo/model1/model1_results_gpu'
    checkpoint_root='./checkpoints/ppo/model1_gpu'
elif model == 'model2':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [16, 16], 8],[32, [4, 4], 2],[256, [11, 11], 1]]
    save_file = './training_results/ppo/model2/model2_results_gpu'
    checkpoint_root='./checkpoints/ppo/model2_gpu'
elif model == 'model3':
    config['model']['dim'] = 252
    config['model']['conv_filters'] = [[16, [8, 8], 4],[16, [8, 8], 4], [32, [4, 4], 2], [256, [8, 8], 1]]
    save_file = './training_results/ppo/model3/model3_results_gpu'
    checkpoint_root='./checkpoints/ppo/model3_gpu'
elif model == 'model4':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2],[32, [4, 4], 2], [256, [11, 11], 1]]
    save_file = './training_results/ppo/model4/model4_results_gpu'
    checkpoint_root='./checkpoints/ppo/model4_gpu'
elif model == 'model5':
    config['model']['dim'] = 252
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2], [32, [4, 4], 2], [256, [16, 16], 1]]
    save_file = './training_results/ppo/model5/model5_results_gpu'
    checkpoint_root='./checkpoints/ppo/model5_gpu'
elif model == 'model6':
    config['model']['dim'] = 168
    config['model']['conv_filters'] = [[16, [8, 8], 4],[32, [4, 4], 2],[256, [21, 21], 1]]
    save_file = './training_results/ppo/model6/model6_results_gpu'
    checkpoint_root='./checkpoints/ppo/model6_gpu'

agent = ppo.PPOTrainer(config, env='Pong-v0')
policy=agent.get_policy()
print(policy.model.model_config)
print(policy.model.base_model.summary())

print("Configuración del agente:\n\n" + str(config))
print("\nConfiguración del modelo del agente:\n\n" + str(config["model"]))

t0 = time.time()
n_iter = int(sys.argv[3])
training.full_train(checkpoint_root, agent, n_iter, save_file)
t1 = time.time()-t0
print("Total time for the " + str(n_iter) + " training iterations: " + str(t1))