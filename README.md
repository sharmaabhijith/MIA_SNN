#  🧠🔒 MIA_SNN

**MIA_SNN** is a project focused on implementing and analyzing *Membership Inference Attacks (MIA)* on *Spiking Neural Networks (SNNs)*. The codebase includes tools for computing thresholds for latency (T=1 to T=4) and calibrating converted SNN models. All models are trained using the **PyTorch** framework.

## 📋 Table of Contents

- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
  - [Set Up the Python Environment](#set-up-the-python-environment)
  - [Install Dependencies](#install-dependencies)
  - [C Routine Setup](#c-routine-setup)
- [🚀 Usage](#-usage)
  - [🛠️ Training an ANN Model](#️-training-an-ann-model)
  - [📊 Computing Thresholds](#-computing-thresholds)
  - [🔄 SNN Model Calibration](#-snn-model-calibration)
  - [🕵️‍♂️ Running Membership Inference Attacks](#️-running-membership-inference-attacks)
  - [📜 Utilizing Bash Scripts for Experiments](#-utilizing-bash-scripts-for-experiments)
- [📓 Demo Notebook](#-demo-notebook)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)

## ✨ Features

- 🏋️‍♂️ Training *Artificial Neural Network (ANN)* models on **CIFAR-10** and **CIFAR-100** datasets
- 🔄 Converting trained ANN models to *Spiking Neural Networks (SNNs)*
- 📏 Computing thresholds for latency (T=1 to T=4) and calibrating SNN models
- 🕵️‍♂️ Implementing three *Membership Inference Attacks (MIA)* RMIA, Attack P and Attack R to assess the privacy of SNNs

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

### C Routine Setup

The project uses a C routine to accelerate computation speed. The `test.so` file is included in the repository. If you encounter any errors loading the `.so` file, you'll need to compile `test.c` to generate a new `.so` file.

## 🚀 Usage

Our SNN training code is based and inspired out of this repo (Data Driven Threshold and Potential Initialization for Spiking Neural
Networks)[https://github.com/srinuvaasu/data_driven_init]

### 🛠️ Training an ANN Model

You need to train ANN model for SNN conversion using `train_ann.py`:

```bash
python3 train_ann.py --dataset [cifar10|cifar100] --model [vgg16|resnet18|resnet20|cifarnet] --reference_models 4
```

### 📊 Computing Thresholds

To compute the threshold and initial potential values:

```bash
python3 feature_extraction.py --iter 1 --samples 100 --model [vgg16|resnet18|resnet20|cifarnet] --dataset [cifar10|cifar100] --checkpoint dir-name --reference_models 4
```

Parameters:
- `iter`: Number of iterations required to find the membrane potential and initial potential values
- `samples`: Number of training data points used for computing the optimal threshold and initial potential values
- `dir-name`: Path to the directory where the trained models are stored

### 🔄 SNN Model Calibration

To calibrate the converted SNN model:

```bash
python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --reference_models 4
```

Training specifications:
- For t=1: Trains an SNN model initialized with the weights of ANN model (50 epochs recommended)
- For t>1: Trains an SNN model initialized with the weights of SNN model with latency t-1 (30 epochs recommended)

### 🕵️‍♂️ Running Membership Inference Attacks

Our MIA attack code and logic is inspired out of this work: (Low-Cost High-Power Membership Inference Attacks)[https://arxiv.org/pdf/2312.03262]
To perform a Membership Inference Attack on the trained SNN model:

```bash
python3 attack.py --attack [attack_p|attack_r|rmia] --dataset [cifar10|cifar100] --model [vgg16|resnet18|resnet20|cifarnet] --model_type [ann|snn] --t 1 --reference_models 4
```

## 📜 Utilizing Bash Scripts for Experiments

The `Scripts` directory contains various Bash scripts designed to streamline the experimental workflow of the MIA_SNN project. Here's how to effectively use them:

**Navigate to the `Scripts` Directory**:
   - Open your terminal and changes script configuration based on your requirement
     ```bash
     # Train ANN
     bash Scripts/01_train_ann.sh
     # Compute Threshold
     bash Scripts/02_threshold.sh
     # Convert ANN to SNN
     bash Scripts/03_calibrate.sh
     # Perform Attack
     bash Scripts/04_attacks.sh
     ```

## 📓 Demo Notebook

For a hands-on demonstration of the MIA attack on trained_SNN models, you can explore the `demo.ipynb` notebook located in the `notebooks/` directory.

## 📂 Project Structure

```
📂 MIA_SNN
├── Attacks/        # Contains scripts related to implementing various attack strategies
├── GradCAM/        # Tools for visualizing model decisions using Grad-CAM
├── Helpers/        # Utility functions for data processing and model management
├── Models/         # Definitions of various neural network architectures
├── Preprocess/     # Scripts for data preprocessing and augmentation
├── Scripts/        # Miscellaneous scripts for evaluation and analysis
├── notebooks/      # Jupyter notebooks for exploratory analysis and demonstrations
├── saved_models/   # Directory to save and load trained model checkpoints
├── test.c         # C routine for accelerating computation
├── test.so        # Compiled shared object file for C routine
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
