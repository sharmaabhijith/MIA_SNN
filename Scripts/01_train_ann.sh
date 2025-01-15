#! /bin/bash


#SBATCH --job-name=ann_model_traning # Job name
#SBATCH --output=outputs/ann_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet34")

# Dataset
DATASETS=("cifar10")

REF_MODELS=(4)

# Loop through each model and run the training script
for REF in "${REF_MODELS[@]}"; do
	for DATASET in "${DATASETS[@]}"; do
		for MODEL in "${MODELS[@]}"; do
			echo "Training $MODEL on $DATASET..."
			python3 train_ann.py --dataset $DATASET --model $MODEL --reference_models $REF
			echo "Finished training $MODEL"
			echo "--------------------------------"
		done
		echo "Finished training all models on $DATASET"
			echo "================================="
	done
	echo "All models trained successfully"
done
