import argparse
import time

from PIL import Image

import tflite_runtime.interpreter as tflite
import numpy as np
import platform

import ray.rllib.env.atari_wrappers as wrappers
import gym

from statistics import mean
import csv


def make_interpreter(model_file):
  return tflite.Interpreter(model_path=model_file)

def keep_going(steps, num_steps, episodes, num_episodes):
  if num_episodes:
    return episodes < num_episodes
  if num_steps:
    return steps < num_steps
  return True

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=False, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-s', '--steps', type=int, default=10000,
      help='Number of times to run inference (overwriten by --episodes')
  parser.add_argument(
      '-e', '--episodes', type=int, default=0,
      help='Number of complete episodes to run (overrides --steps)')
  parser.add_argument(
      '-o', '--output', default = None,
      help= 'CSV file to store timing results')
  args = parser.parse_args()

  # Create TFLite interpreter
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print('Input details: ', input_details)
  print('Output details: ', output_details)

  # Get image dim
  dim = input_details[0]['shape'][1]

  # Create env
  env = wrappers.wrap_deepmind(gym.make('Pong-v0'), dim = dim)
  
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')

  steps=0
  episodes = 0
  timing_results=[]
  while keep_going(steps, args.steps, episodes, args.episodes):
    this_episode_timing_results ={}
    this_episode_timing_results['episode'] = episodes
    reward_total=0.0
    done = False

    image = env.reset()
    image = image[np.newaxis, ...]


    if input_details[0]['dtype'] == np.float32:
      image=np.float32(image)
    if input_details[0]['dtype'] == np.uint8:
      image=np.uint8(image)

    episode_times = []
    steps_this_episode = 0
    interpreter.set_tensor(input_details[0]['index'], image)
    while not done and keep_going(steps, args.steps, episodes, args.episodes):

      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      episode_times.append(inference_time)

      output_data = interpreter.get_tensor(output_details[0]['index'])

      action = np.argmax(output_data)
      
      # Step environment and get reward and done information
      image, reward, done, _ = env.step(action)

      # Place new image as the new model's input
      image = image[np.newaxis, ...]
      
      if input_details[0]['dtype'] == np.float32:
        image=np.float32(image)
      if input_details[0]['dtype'] == np.uint8:
        image=np.uint8(image)
      

      interpreter.set_tensor(input_details[0]['index'], image)

      # Get cummulative episode reward
      reward_total+=reward

      steps+=1
      steps_this_episode+=1
    

    if done:
      episodes +=1

    step_time_ms = mean(episode_times[1:])*1000
    print("Episode {}, Reward: {}, Mean step time: {:.2f}".format(episodes, reward_total, step_time_ms))
    this_episode_timing_results['step_time_ms'] = step_time_ms
    this_episode_timing_results['steps'] = steps_this_episode
    this_episode_timing_results['reward'] = reward_total
    timing_results.append(this_episode_timing_results)
    print('-----------------------')
  
  if args.output is not None:
    with open(args.output, mode='w') as f:
      fieldnames = list(timing_results[0].keys())
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for row in timing_results:
        writer.writerow(row)

if __name__ == '__main__':
  main()
