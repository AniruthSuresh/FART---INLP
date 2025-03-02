FART-INLP
=========

Repository for maintaining all codes related to the INLP Project.

This repository contains the implementation of [FFT-Net](https://arxiv.org/pdf/2502.18394) , an adaptive spectral filtering framework that leverages the Fast Fourier Transform (FFT) to achieve global token mixing in **O(n log n)** time. By transforming inputs into the frequency domain, FFT-Net **exploits the orthogonality and energy preservation guaranteed by Parseval’s theorem** to efficiently capture long-range dependencies.

Installation and Dependencies
-----------------------------

Run the following commands to set up the environment:


```
sudo apt update
sudo apt install tmux -y
```

Running the Code
----------------

## 1. CIFAR Experiment

The ``cifar/`` directory in this repository is dedicated to training and comparing two neural network models on the CIFAR-10 dataset:

- **FFTNetViT** – A Vision Transformer variant that uses Fast Fourier Transform (FFT)-based attention mechanisms.
- **Vision Transformer (ViT)** – A standard Vision Transformer implementation.


Steps to Run:


```
cd cifar

python cifar-updated.py  # Trains the model and saves validation metrics to a text file
python plot.py   # Generates performance plots
```



