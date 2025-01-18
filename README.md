# Deep Learning Implementations

A collection of deep learning implementations using different frameworks (PyTorch, TensorFlow, PyTorch Lightning) for computer vision tasks.

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-yellow.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)

## 📋 Projects

### MNIST CNN Implementation
- TensorFlow implementation (`tf_mnist_cnn.py`)
- PyTorch implementation (`pytorch_mnist_cnn.py`)
- Both implementations include:
  - Convolutional Neural Network architecture
  - Training visualization
  - Model evaluation
  - Performance metrics

### Fashion MNIST CNN
Two different approaches to demonstrate regularization techniques:
1. Without Regularization (`pytorch_fmnist_cnn_w_no_regularization.py`)
2. With Regularization (`pytorch_fmnist_cnn_w_regularization.py`)

Regularization techniques implemented:
- Data Augmentation
- Dropout
- Batch Normalization
- L2 Regularization

### PyTorch Lightning CNN
Modern implementation using PyTorch Lightning framework featuring:
- Custom Dataset implementation
- Advanced training features:
  - Early stopping
  - Model checkpointing
  - TensorBoard logging
  - Custom callbacks
- GPU acceleration
- Model deployment preparation

## 🏗️ Model Architectures

### Basic CNN Architecture
```
Input -> Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPool2D -> 
Flatten -> Dense -> Dense(output)
```

### Regularized CNN Architecture
```
Input -> Conv2D -> BatchNorm -> ReLU -> Dropout ->
Conv2D -> BatchNorm -> ReLU -> Dropout -> MaxPool2D ->
Flatten -> Dense -> Output
```

## 🚀 Features

- Multiple framework implementations
- Training visualization
- Performance metrics
- Model saving and loading
- GPU support
- Data augmentation
- Various regularization techniques
- TensorBoard integration

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Btzel/computer-vision-deeplearning.git
cd computer-vision-deeplearning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision tensorflow pytorch-lightning
```

## 💡 Usage

### TensorFlow MNIST:
```bash
python tf_mnist_cnn.py
```

### PyTorch MNIST:
```bash
python pytorch_mnist_cnn.py
```

### Fashion MNIST:
```bash
# Without regularization
python pytorch_fmnist_cnn_w_no_regularization.py

# With regularization
python pytorch_fmnist_cnn_w_regularization.py
```

### PyTorch Lightning:
```bash
python pytorch_lightning_cnn.py
```

## 📊 Results

Each implementation includes:
- Training/validation loss curves
- Accuracy metrics
- Confusion matrices (where applicable)
- Per-class accuracy analysis

## 📁 Project Structure

```
computer-vision-deeplearning/
├── tf_mnist_cnn.py              # TensorFlow MNIST implementation
├── pytorch_mnist_cnn.py         # PyTorch MNIST implementation
├── pytorch_fmnist_cnn_w_no_regularization.py  # Basic Fashion MNIST
├── pytorch_fmnist_cnn_w_regularization.py     # Regularized Fashion MNIST
└── pytorch_lightning_cnn.py     # PyTorch Lightning implementation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** This repository is for educational purposes and demonstrates different approaches to implementing CNNs using various deep learning frameworks.
