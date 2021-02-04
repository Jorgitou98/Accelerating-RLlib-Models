import ray
import ray.rllib.agents.ppo as ppo
import json, os, shutil, sys

ray.shutdown()
ray.init(ignore_reinit_error=True)

print("Dashboard URL: http://{}".format(ray.get_webui_url()))

CHECKPOINT_ROOT = "/tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

results = []
episode_data = []
episode_json = []

for n in range(N_ITER):
    result = agent.train()
    results.append(result)
    episode = {'n': n,
               'episode_reward_min': result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max': result['episode_reward_max'],
               'episode_len_mean': result['episode_len_mean']}
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = agent.save(CHECKPOINT_ROOT)
    print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))


   # print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')

#for n in range(N_ITER):
  #result = agent.train()
  #file_name = agent.save(CHECKPOINT_ROOT)
#
  #print(s.format(
    #n + 1,
    #result["episode_reward_min"],
    #result["episode_reward_mean"],
    #result["episode_reward_max"],
    #result["episode_len_mean"],
    #file_name
   #))


policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())
