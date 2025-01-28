#! /bin/bash

#SBATCH --job-name=ann_model_traning1 # Job name
#SBATCH --output=outputs/ann_train_output1.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet34")

# Dataset
DATASETS=("cifar100")

# Number of reference models in the experiment
REF_MODELS=(4)

# Loop through each model and run the training script
for REF in "${REF_MODELS[@]}"; do
	for DATASET in "${DATASETS[@]}"; do
		for MODEL in "${MODELS[@]}"; do
			echo "Training $MODEL on $DATASET..."
			python3 train_ann.py --dataset $DATASET --model $MODEL --reference_models $REF --epochs 100
            python3 feature_extraction.py --iter 4 --sample 10000 --dataset "$DATASET" --model "$MODEL" --reference_models $REF
			echo "Finished training $MODEL"
			echo "--------------------------------"
		done
		echo "Finished training all models on $DATASET"
			echo "================================="
	done
	echo "All models trained successfully"
done