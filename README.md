## DenseNet for Neuromorphic Datasets

This repository contains the code for training a 2D-CNN with DenseNet architecture on both
the (Neuromorphic) N-MNIST and the (Fixational Neuromorphic) FN-MNIST datasets.
The aim is to benchmark the spatial information of such event-based data. Two different scripts are available,
for using either grayscale frames (1 channel) or both ON/OFF polarities (2 channels) accumulated from the events.
Unfortunately data cannot be uploaded due to its size. Finally, a Dockerfile is available for building a 
docker image (based on the tensorflow/tensorflow:latest-gpu image) in order to run the scripts from the relative 
docker container (see the file in the _docs_ folder for further info).


### 1 - Training with two-channeled frames

#### From N-MNIST:
```bash
python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/N-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20
```

#### From FN-MNIST:
```bash
python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/FN-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20
```


### 2 - Training with grayscale frames

#### From N-MNIST:
```bash
python3 /home/cnn2d/code/densenet-grayscale.py  --data_path /home/cnn2d/data/N-MNIST  --log_path /home/cnn2d/data/logs_gray/N-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20
```

#### From FN-MNIST:
```bash
python3 /home/cnn2d/code/densenet-grayscale.py  --data_path /home/cnn2d/data/FN-MNIST  --log_path /home/cnn2d/data/logs_gray/FN-MNIST  --random_seed 0  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20
```
