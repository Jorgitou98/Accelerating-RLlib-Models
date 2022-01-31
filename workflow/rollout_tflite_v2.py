import argparse
import time

from PIL import Image

import tflite_runtime.interpreter as tflite
import numpy as np
import platform

#import ray.rllib.env.atari_wrappers as wrappers
import ray.rllib.env.wrappers.atari_wrappers as wrappers
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
  #env = wrappers.wrap_deepmind(gym.make('Taxi-v3'), dim = dim)
  env = gym.make('Taxi-v3')
  #env = wrappers.wrap_deepmind(env)
  
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')

  steps=0
  episodes = 0
  timing_results=[]

  done = False

  image = env.reset()

  #image = image[np.newaxis, ...]
  print(image)
  image = np.array([int(i == image) for i in range(500)])
  print(image)
  #print(image)

  if input_details[0]['dtype'] == np.float32:
    image=np.float32(image)
  if input_details[0]['dtype'] == np.uint8:
    image=np.uint8(image)

  interpreter.set_tensor(input_details[0]['index'], [image])

  this_step = 0

  while not done and keep_going(steps, args.steps, episodes, args.episodes):
    env.render()

    input("Press to continue...")

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[1]['index'])
    print(output_data)

    action = np.argmax(output_data)
    #print(action)
      
    # Step environment and get reward and done information
    image, reward, done, prob = env.step(action)
    print("Step {} --- Applied action {}. Returned observation: {}. Returned reward: {}. Probability: {}".format( this_step, action, image, reward, prob["prob"] ))
    this_step = this_step+1

    # Place new image as the new model's input
    #image = image[np.newaxis, ...]
    image = np.array([int(i == image) for i in range(500)])
    print(image)
      
    if input_details[0]['dtype'] == np.float32:
      image=np.float32(image)
    if input_details[0]['dtype'] == np.uint8:
      image=np.uint8(image)
      
    interpreter.set_tensor(input_details[0]['index'], [image])

if __name__ == '__main__':
  main()
