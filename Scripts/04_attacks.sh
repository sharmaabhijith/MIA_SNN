#! /bin/bash

#SBATCH --job-name=snn_calibration # Job name
#SBATCH --output=outputs/snn_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos

ATTACK="rmia"
DATASET="cifar100"
MODEL="resnet18"
MODEL_TYPE="ann"
LATENCY=1
REF_MODELS=4
CALIBRATION=0
DROPOUT=0.01
N_SAMPLES=10

# Loop through each model and run the training script
python3 attack.py --attack $ATTACK --dataset $DATASET --model $MODEL --model_type $MODEL_TYPE --t $LATENCY --calibration $CALIBRATION --dropout $DROPOUT --n_samples $N_SAMPLES --reference_models $REF_MODELS
