# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.

   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.

   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --input images/parrot.jpg
   ```
"""

import argparse
import time

from PIL import Image

import tflite_runtime.interpreter as tflite
import numpy as np
import platform

import ray.rllib.env.atari_wrappers as wrappers
import gym

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


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
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  # Create TFLite interpreter
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  #image=np.random.rand(1,252,252,4) * 255

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print('Input details: ', input_details)
  print('Output details: ', output_details)

  # Get image dim
  dim = input_details[0]['shape'][1]

  # Create env
  env = wrappers.wrap_deepmind(gym.make('Pong-v0'), dim = dim)

  '''
  image=np.load( "../datasets/dataset_model3.npy" )
  image = image[np.newaxis, ...]
  '''

  image = env.reset()
  image = image[np.newaxis, ...]

  #print(image)
  print('Images shape: ', image.shape)

  if input_details[0]['dtype'] == np.float32:
    image=np.float32(image)
  if input_details[0]['dtype'] == np.uint8:
    image=np.uint8(image)

  interpreter.set_tensor(input_details[0]['index'], image)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start

    print('---- output[0] ----')
    output_data = interpreter.get_tensor(output_details[0]['index'])

    action = np.argmax(output_data)

    if output_details[0]['dtype'] == np.uint8:
      print("INT8 DATA")
      #print(output_data)
      scale, zero_point = output_details[0]['quantization']
      #print(scale, zero_point)
      print( scale * ( np.float32(output_data) - zero_point ) )
    else:
      print("FLOAT DATA")
      print(output_data)

    print('---- output[1] ----')
    output_data = interpreter.get_tensor(output_details[1]['index'])

    if output_details[1]['dtype'] == np.uint8:
      print("INT8 DATA")
      #print(output_data)
      scale, zero_point = output_details[1]['quantization']
      #print(scale, zero_point)
      print( scale * ( np.float32(output_data) - zero_point ) )
    else:
      print("FLOAT DATA")
      print(output_data)

    print('---- end ----')

    print('%.1fms' % (inference_time * 1000))

    image, reward, done, info = env.step(action)
    image = image[np.newaxis, ...]
    print("Reward: ", reward)

  print('-------RESULTS--------')

if __name__ == '__main__':
  main()
