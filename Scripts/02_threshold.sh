#! /bin/bash

#SBATCH --job-name=snn_threshold_calculation
#SBATCH --output=outputs/threshold_calc_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet34")

# Dataset
DATASETS=("cifar10")

# Number of reference models in the experiment
REF_MODELS=(4)

# Loop through each model and run the training script
for REF in "${REF_MODELS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
                for MODEL in "${MODELS[@]}"; do
                        echo "Feature extraction of $MODEL on $DATASET..."
                        python3 feature_extraction.py --iter 4 --sample 10000 --dataset "$DATASET" --model "$MODEL" --reference_models $REF
                        echo "Finished feature extraction of $MODEL"
                        echo "--------------------------------"
                done
                echo "Finished feature extraction of all models on $DATASET"
                echo "================================="
        done
        echo "All models trained successfully"
done
