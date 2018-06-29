# Semantic Segmentation of Cancer Cells
PyTorch implementation of several CNN-based models for segmentation of cancer cells.
# Requirement
Python 3.6.4

PyTorch 0.4.0

Some other libraries: NumPy 1.14.0, SciPy 1.0.0, Matpotlib 2.1.2

I recommend installation of [PyTorch](https://pytorch.org/) with CUDA using [Anaconda](https://anaconda.org/), which includes most of the libraries required. For example, Linux users can run the following command in the terminal:
```
conda install pytorch torchvision cuda91 -c pytorch
```

(This is my environment, but others may also work)
# Model
The first model is a modified version of the model in [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/abs/1505.04366/).

Network configuration:

| Layer | Output Size ![](https://latex.codecogs.com/gif.latex?C%5Ctimes%20H%5Ctimes%20W) | Activations | Weights |
| ------------- |:-------------:| -----:| -----:|
| input | ![](https://latex.codecogs.com/gif.latex?3%5Ctimes%20320%5Ctimes%20320) | 307200 | 0 |
| conv1-1 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6553600 | 1728 |
| conv1-2 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6553600 | 36864 |
