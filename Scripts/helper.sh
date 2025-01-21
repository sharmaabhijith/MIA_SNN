#! /bin/bash

#SBATCH --job-name=ann_model_traning1 # Job name
#SBATCH --output=outputs/ann_train_output1.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


python 