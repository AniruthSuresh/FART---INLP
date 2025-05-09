## Fine-tuning BART and FNet-BART Models

In this branch, we fine-tune both the base BART model and a modified BART model where the attention mechanism is replaced with FNet-style Fourier transforms. Experiments are conducted on the following datasets:

### 1. SST-2 Dataset

To run the base BART model:

```bash
cd sst-2
python3 BART_SST.py
```

To run the FNet-BART model:

```bash
cd sst-2
python3 FnetBART_SST.py
```

### 2. SWAG Dataset

To run the base BART model:

``` bash
cd swag
python3 BART_SWAG.py
```

To run the FNet-BART model:

``` bash
cd swag
python3 FnetBART_SWAG.py
```

### Adaptive Filtering with FFTNet

Additionally, we experiment with adaptive filtering techniques inspired by the [FFTNet paper](https://arxiv.org/pdf/2502.18394v1). Here, we replace the attention mechanism with Fourier filters and benchmark the model on the SST-2 dataset.

To run the FFTNet-based BART model:

```bash 
cd fftNet-spectre
python3 fftnet_bart_sst2.py
```


