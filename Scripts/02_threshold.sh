#! /bin/bash


#SBATCH --job-name=snn_threshold_calculation
#SBATCH --output=../outputs/threshold_calc_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("vgg16" "resnet18")

# Dataset
DATASETS=("cifar10")

# Loop through each model and run the training script
for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
                echo "Feature extraction of $MODEL on $DATASET..."
                python3 ../feature_extraction.py --iter 4 --sample 10000 --dataset "$DATASET" --model "$MODEL"
                echo "Finished feature extraction of $MODEL"
                echo "--------------------------------"
        done
	echo "Finished feature extraction of all models on $DATASET"
        echo "================================="
done
echo "All models trained successfully"
