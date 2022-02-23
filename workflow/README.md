1. Train for Taxi

```sh
python train_ppo_taxi.py --model=1 --gpu=gpu0 --workers=1 --save-name=model1_taxi --iters=10
```

2. Export model

```sh
python model_saver.py checkpoints/ppo/model1_taxi/checkpoint_000009/checkpoint-9 exported_models/checkpoint-9
```

3. Convert model

```sh
python tflite_converter_pb.py exported_models/checkpoint-9 exported_models/taxi_9.tflite
```

4. Manual rollout (TFLITE)

```sh
python rollout_tflite.py -m exported_models/taxi_9.tflite -s 10 -e 10
```

5. Manual rollout (using RLlib). For reference

```sh
python rollout_with_time.py checkpoints/ppo/model1_taxi/checkpoint_000009/checkpoint-9 --run=PPO --env=Taxi-v3 --time-output=rollout_results/volta1/model1_no_gpus_0_workers.csv --no-render --gpu=none --episodes=1000 --config='{"num_workers":1, "num_gpus_per_worker":0, "num_gpus":0, "explore":false}'
```
6. Model quantization 8 bits

```sh
python quantizer8b.py dataset_taxi.npy exported_models/checkpoint-90 exported_models/checkpoint-90_quant.tflite
```
7. Model quantization 16 bits

```sh
python quantizer16b.py exported_models/checkpoint-90 exported_models/checkpoint-90_quant.tflite
```
8. Test with consecutive seeds given by episode number (used to compare models)

```sh
python rollout_tflite_sequential_seeds.py --m exported_models/checkpoint-90_quant8_noinput.tflite -e 100
```
