
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
with open('prueba.npy', 'wb') as f:
    for _ in range(101):
        np.save(f, obs)
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)


ray.shutdown()


#open(tflite_dir, "wb").write(tflite_model)




