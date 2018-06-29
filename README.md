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
$y=\frac{1}{x}$
| Layer | C$$\times$$H | Activations | Weights |
| ------------- |:-------------:| -----:| -----:|
| input | 3$$\times$$5 | 307200 | 0 |
