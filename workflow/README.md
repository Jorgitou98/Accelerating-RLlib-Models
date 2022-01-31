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
