#! /bin/bash


#SBATCH --job-name=snn_calibration # Job name
#SBATCH --output=outputs/snn_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet34")

# Dataset
DATASETS=("cifar10")

# Latency
LATENCY=$(seq 2 4)

TRAIN_TEST_SPLIT=0.5

EXP_TYPE="RMIA"

# Loop through each model and run the training script
for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
		echo "Training $MODEL on $DATASET..."
		echo "Training for T = 1"
		python3 train_snn_converted.py --t 1 --epochs 50 --dataset "$DATASET" --model "$MODEL" --train_split $TRAIN_TEST_SPLIT --exp_type $EXP_TYPE
		echo  "Training finished for T = 1"
		for T in $LATENCY; do
                	echo "Training for T = $T"
                	python3 train_snn_converted.py --t $T --epochs 30 --dataset "$DATASET" --model "$MODEL" --train_split $TRAIN_TEST_SPLIT --exp_type $EXP_TYPE
			echo "Training finished for T = $T"
		done
		echo "Finished training $MODEL"
        	echo "--------------------------------"
	done
	echo "Finished training all models on $DATASET"
        echo "================================="
done
echo "All models trained successfully"
