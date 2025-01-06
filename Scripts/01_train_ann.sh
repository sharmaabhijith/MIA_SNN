#! /bin/bash


#SBATCH --job-name=ann_model_traning # Job name
#SBATCH --output=outputs/ann_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet18" "vgg16" "resnet34")

# Dataset
DATASETS=("cifar10")

TRAIN_TEST_SPLIT=0.5

# Loop through each model and run the training script
for DATASET in "${DATASETS[@]}"; do
	for MODEL in "${MODELS[@]}"; do 
		echo "Training $MODEL on $DATASET..."
		python3 train_ann.py --dataset "$DATASET" --model "$MODEL" --train_split $TRAIN_TEST_SPLIT
		echo "Finished training $MODEL"
		echo "--------------------------------"
	done
	echo "Finished training all models on $DATASET"
        echo "================================="
done
echo "All models trained successfully"
