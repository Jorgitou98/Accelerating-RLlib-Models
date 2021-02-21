import ray
import time
import os

ray.shutdown()
restore_path = input("Enter the restore checkpoint path: ")
num_episodes = int(input("Enter the number of episodes to run: "))
output_dir = input("Enter the output file path: ")
output_pkl = output_dir + '.pkl'
output_csv = output_dir + '.csv'
t0 = time.time()
os.system("python rollout.py " + restore_path + " --env=Pong-v0 --run=PPO --episodes=" + str(num_episodes) + " --out=" + output_pkl + " --time-output=" + output_csv + " --save-info --use-shelve")
t1 = time.time() - t0
print("Rollout total time: " + str(t1))