import ray
import time
import os
import sys

ray.shutdown()
restore_path = sys.argv[1]
num_episodes = sys.argv[2]
output_dir = sys.argv[3]
config = sys.argv[4]
output_pkl = output_dir + '.pkl'
output_csv = output_dir + '.csv'
print(config)
print("python rollout.py " + restore_path + " --env=Pong-v0 --run=PPO --episodes=" + str(num_episodes) + " --out=" + output_pkl + " --time-output=" + output_csv + " --save-info --use-shelve --no-render --config '" + config + "'")
t0 = time.time()
os.system("python rollout.py " + restore_path + " --env=Pong-v0 --run=PPO --episodes=" + str(num_episodes) + " --out=" + output_pkl + " --time-output=" + output_csv + " --save-info --use-shelve --no-render --config '" + config + "'")
t1 = time.time() - t0
print("Rollout total time: " + str(t1))