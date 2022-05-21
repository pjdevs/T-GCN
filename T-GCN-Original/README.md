# T-GCN

This a custom version of [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch) suited for our tests.

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0

## Model Training

```bash
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 100 --loss mse_with_regularizer --gpus 1 --data pemsbay
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.
