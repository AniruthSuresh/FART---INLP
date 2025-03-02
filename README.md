FART-INLP
=========

Repository for maintaining all codes related to the INLP Project.

This repository contains the implementation of [FFT-Net](https://arxiv.org/pdf/2502.18394) , an adaptive spectral filtering framework that leverages the Fast Fourier Transform (FFT) to achieve global token mixing in **O(n log n)** time. By transforming inputs into the frequency domain, FFT-Net **exploits the orthogonality and energy preservation guaranteed by Parseval’s theorem** to efficiently capture long-range dependencies.

Installation and Dependencies
-----------------------------


Run the following commands to set up the environment:


```

conda env create -f env.yml
conda activate inlp  # Replace 'inlp' with the environment name in env.yml

sudo apt update
sudo apt install tmux -y
pip install wandb

```

Running the Code
----------------

## 1. CIFAR Experiment

Load the dataset from [CIFAR-FftNet](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/aryan_garg_students_iiit_ac_in/Ekb0vZ4KgSlMsTcsKG60uhwBmVAILoBiyJdhjx26OdC_IQ?e=H65GHn) and store it in the `/data` folder.

You could also just run `cifar-updated.py` without the above step. It's just quicker to download from OneDrive than from the internet.

The `cifar/` directory in this repository is dedicated to training and comparing two neural network models on the CIFAR-10 dataset:

- **FFTNetViT** – A Vision Transformer variant that uses Fast Fourier Transform (FFT)-based attention mechanisms.
- **Vision Transformer (ViT)** – A standard Vision Transformer implementation.

### Steps to Run:


```
cd cifar

python cifar-updated.py  # Trains the model and saves validation metrics to a text file
python plot.py   # Generates performance plots
```


After training on the CIFAR dataset, the best model weights can be found [here](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/aryan_garg_students_iiit_ac_in/Ekb0vZ4KgSlMsTcsKG60uhwBmVAILoBiyJdhjx26OdC_IQ?e=GZIOBX)


## TODO: Model Performance Visualization

Extract the accuracy and loss from wandb and the timing for each epochs and the total training time from the result txt files and then visualize it for the final presentation !!
