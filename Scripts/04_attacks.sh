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

# Define the parameter combinations
ATTACKS=("attack_r" "rmia")
DATASETS=("cifar10" "cifar100")
MODEL="resnet18"
MODEL_TYPE="{'model_0':'snn', 'model_1':'ann', 'model_2':'ann', 'model_3':'ann', 'model_4':'ann'}"
LATENCIES=(1 2 4)
REF_MODELS=4
CALIBRATIONS=(0 1)
DROPOUT=0.01
N_SAMPLES=5

# Iterate over all combinations
for ATTACK in "${ATTACKS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for LATENCY in "${LATENCIES[@]}"; do
            for CALIBRATION in "${CALIBRATIONS[@]}"; do
                echo "Running attack: $ATTACK, dataset: $DATASET, latency: $LATENCY, calibration: $CALIBRATION"

                # Execute Python script with the current parameter combination
                python3 attack.py \
                    --attack "$ATTACK" \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --model_type "$MODEL_TYPE" \
                    --t "$LATENCY" \
                    --calibration "$CALIBRATION" \
                    --dropout "$DROPOUT" \
                    --n_samples "$N_SAMPLES" \
                    --reference_models "$REF_MODELS"

            done
        done
    done
done