import ray
import ray.rllib.agents.ppo as ppo
import json, os, shutil, sys
import gym
import pprint
import time
import shelve
from tensorflow import keras
from ray import tune
import csv

@ray.remote(num_cpus=0, num_gpus=1)
def full_train(checkpoint_root, agent, n_iter, save_file, n_ini = 0, header = True, restore = False, restore_dir = None):
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} learn_time_ms {:6.2f} saved {}"
    if(restore):
        if restore_dir == None:
            print("Error: you must specify a restore path")
            return
        agent.restore(restore_dir)
    #else:
        #shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)
    results = []
    episode_data = []
    episode_json = []

    total_learn_time = 0
    for n in range(n_iter):
        result = agent.train()
        results.append(result)
        episode = {'n': n_ini + n + 1,
                   'episode_reward_min': result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max': result['episode_reward_max'],
                   'episode_len_mean': result['episode_len_mean'],
                   'learn_time_ms': result['timers']['learn_time_ms']
                   }
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = agent.save(checkpoint_root)
        print(s.format(
        n_ini + n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        result["timers"]["learn_time_ms"],
        file_name
       ))
        total_learn_time+= result["timers"]["learn_time_ms"]

    print("Total learn time: " + str(total_learn_time))
    print("Average learn time per iteration: " + str(total_learn_time/n_iter))

    with open(save_file + '.json', mode='a') as outfile:
        json.dump(episode_json, outfile)

    with open(save_file + '.csv', mode='a') as csv_file:
        fieldnames = ['n', 'episode_reward_min', 'episode_reward_mean', 'episode_reward_max', 'episode_len_mean', 'learn_time_ms']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        for row in episode_data:
            writer.writerow(row)