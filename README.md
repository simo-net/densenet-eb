## DenseNet for Neuromorphic Datasets

This repository contains the code for training a 2D-CNN with DenseNet architecture on both
the (Neuromorphic) N-MNIST and the (Fixational Neuromorphic) FN-MNIST for benchmarking the
spatial information in them. Two different scripts are available, for using either grayscale
frames (1 channel) or both ON/OFF polarities (2 channels).


### 1 - Training with two-channeled frames

#### From N-MNIST:
```bash
python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/N-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --store_model_info
```

#### From FN-MNIST:
```bash
python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/FN-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --store_model_info
```


### 2 - Training with grayscale frames

#### From N-MNIST:
```bash
python3 /home/cnn2d/code/densenet-grayscale.py  --data_path /home/cnn2d/data/N-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --store_model_info
```

#### From FN-MNIST:
```bash
python3 /home/cnn2d/code/densenet-grayscale.py  --data_path /home/cnn2d/data/FN-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --store_model_info
```
