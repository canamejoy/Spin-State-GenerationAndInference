# 02 · Models

Deep-regression architectures that map a single-axis ($s_z$) magnetization image to the eight
Hamiltonian parameters $\boldsymbol{\theta}$. **All models are trained from scratch** (no
ImageNet pretraining) under a common protocol (MSE loss, Adam, early stopping / LR scheduling,
90° rotation augmentation where noted).

| Notebook | Architecture | Notes |
|----------|--------------|-------|
| [`cnn_baseline.ipynb`](cnn_baseline.ipynb) | Lightweight CNN (from scratch) | Native $39\times39$ input; reference baseline. |
| [`resnet50.ipynb`](resnet50.ipynb) | ResNet-50 | Input resized to $224\times224\times3$. |
| [`densenet121.ipynb`](densenet121.ipynb) | DenseNet-121 | Input resized to $224\times224\times3$. |
| [`xception.ipynb`](xception.ipynb) | Xception | Depthwise-separable convolutions; best accuracy-vs-size trade-off. |
| [`vision_transformer.ipynb`](vision_transformer.ipynb) | Vision Transformer | Via `keras-hub`; patch + self-attention encoder. |

## Common structure of each notebook

1. Install dependencies and load the dataset (`.npz`).
2. Preprocess: normalize $s_z$ images, scale targets, train/val/test split, augmentation.
3. Build the architecture with a dense regression head predicting the eight parameters.
4. Train and export the model weights + target scaler for use in `03_results/`.

## Framework

TensorFlow / Keras (with `keras-hub` for the Vision Transformer), scikit-learn for splitting,
scaling, and metrics.
