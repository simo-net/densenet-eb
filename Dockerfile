FROM tensorflow/tensorflow:latest-gpu
LABEL Maintainer="simonet"
WORKDIR /home/cnn2d
COPY . ./
RUN pip install -r requirements.txt
