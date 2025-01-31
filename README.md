# 🧠 MIA_SNN

**MIA_SNN** is a project focused on implementing and analyzing *Membership Inference Attacks (MIA)* on *Spiking Neural Networks (SNNs)*. The codebase includes tools for computing thresholds for latency (T=1) and calibrating converted SNN models. All models are trained using the **PyTorch** framework.

## 📋 Table of Contents

- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
  - [🛠️ Training an ANN Model](#️-training-an-ann-model)
  - [🔄 Converting an ANN to an SNN](#-converting-an-ann-to-an-snn)
  - [🕵️‍♂️ Running Membership Inference Attacks](#️-running-membership-inference-attacks)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Features

- 🏋️‍♂️ Training *Artificial Neural Network (ANN)* models on **CIFAR-10** and **CIFAR-100** datasets
- 🔄 Converting trained ANN models to *Spiking Neural Networks (SNNs)*
- 📏 Computing thresholds for latency (T=1) and calibrating SNN models
- 🕵️‍♂️ Implementing *Membership Inference Attacks (MIA)* to assess the privacy of SNNs

## ⚙️ Installation

### Set Up the Python Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Ensure you have PyTorch installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```

Note: The requirements.txt file should list all necessary packages, including PyTorch. If it's missing or incomplete, you'll need to install PyTorch separately. Refer to the PyTorch official website for installation instructions tailored to your system.

## 🚀 Usage

### 🛠️ Training an ANN Model

To train an ANN model on either the CIFAR-10 or CIFAR-100 dataset:

```bash
python3 train_ann.py --dataset [cifar10|cifar100] --model [vgg16|resnet18|resnet20|cifarnet]
```

Replace [cifar10|cifar100] with your desired dataset and [vgg16|resnet18|resnet20|cifarnet] with your chosen model architecture.

### 🔄 Converting an ANN to an SNN

After training the ANN model, you can convert it to an SNN:

```bash
python3 train_snn_converted.py --dataset [cifar10|cifar100] --model [vgg16|resnet18|resnet20|cifarnet]
```

This script computes the necessary thresholds and calibrates the SNN model.

### 🕵️‍♂️ Running Membership Inference Attacks

To perform a Membership Inference Attack on the trained SNN model:

```bash
python3 attack.py --dataset [cifar10|cifar100] --model [vgg16|resnet18|resnet20|cifarnet]
```

This will evaluate the privacy vulnerabilities of your SNN model.

## 📂 Project Structure

```
📂 MIA_SNN
├── Attacks/        # Contains scripts related to implementing various attack strategies
├── GradCAM/        # Tools for visualizing model decisions using Grad-CAM
├── Helpers/        # Utility functions for data processing and model management
├── Models/         # Definitions of various neural network architectures
├── Preprocess/     # Scripts for data preprocessing and augmentation
├── Scripts/        # Miscellaneous bash scripts for evaluation and analysis
├── notebooks/      # Jupyter notebooks for exploratory analysis and demonstrations
├── saved_models/   # Directory to save and load trained model checkpoints
└── README.md       # Project documentation
```

## 🤝 Contributing

We welcome contributions to enhance the MIA_SNN project. If you're interested in contributing, please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a new branch for your feature or bug fix
3. 💬 Commit your changes with clear and concise messages
4. 📤 Push your branch to your forked repository
5. 📥 Open a Pull Request detailing your changes and the motivation behind them

Please ensure that your code adheres to the existing style and includes appropriate tests.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.