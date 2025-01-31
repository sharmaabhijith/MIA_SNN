import torch
import pickle
import argparse
import logging
import pandas as pd
import numpy as np
from utils import *
from pathlib import Path
from torch.utils.data import Subset
from Preprocess import get_dataloader_from_dataset, load_dataset
from Attacks.utils import *
from Attacks import *


ATTACK_NAMES = ["attack_p", "attack_r", "rmia"]
DATASET_NAMES = ["cifar10", "cifar100", "imagenette", "imagewoof"]
MODEL_NAMES = ["vgg16", "resnet18", "resnet34"]


# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
# Define arguments for model parameters and settings
parser.add_argument("--attack", default="rmia", type=str, help="Type of MIA attack",
                    choices=ATTACK_NAMES)
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=DATASET_NAMES)
parser.add_argument('--model', default='resnet18', type=str, help='Model name',
                    choices=MODEL_NAMES)
parser.add_argument('--model_type', default='ann', type=str, help='ANN or SNN',
                    choices=["ann", "snn"])
parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory of saved models')
parser.add_argument('--reference_models', default=4, type=int, help='Number of reference models')
parser.add_argument('--result_dir', default='./attack_results', type=str, help='Directory for saving results')
parser.add_argument('--calibration', type=str2bool, default="False", help='Dropout based calibration')
parser.add_argument('--dropout', default=0.01, type=float, help='Image dropout ratio')
parser.add_argument('--n_samples', default=10, type=int, help='Number of samples for calibration')
args = parser.parse_args()

# Check device configuration and set accordingly
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
n_steps = args.t
if args.model_type=="ann":
    half_model_type = "ann"
    full_model_type = "ann"
elif args.model_type=="snn":
    half_model_type = f"snn_T{n_steps}"
    full_model_type = f"ann_snn_T{n_steps}"

n_reference_models = args.reference_models
# Creating directory to save trained models and their logs
primary_model_path = os.path.join(args.checkpoint, args.dataset, args.model, f"ref_models_{n_reference_models}")
primary_result_path = os.path.join(args.result_dir, args.dataset)
primary_log_path = os.path.join("attack_logs", args.dataset, args.model, f"ref_models_{n_reference_models}")
# Create result dir
if os.path.exists(primary_result_path) is False:
    print("Creating result directory:", primary_result_path)
    os.makedirs(primary_result_path)
# Create result csv file if not exists
result_file_path = Path(primary_result_path, f"results_rm{n_reference_models}.csv")
if os.path.exists(result_file_path):
    existing_df = pd.read_csv(result_file_path)
else:
    # Define Multi-level columns
    column_names = DATASET_NAMES
    sub_column_names1 = ATTACK_NAMES
    sub_column_names2 = [f"snn_T{i}" for i in [1, 2, 4]] + ["ann"] 
    column_tuples = [
        (data, attack, model_type) 
        for data in column_names for attack in sub_column_names1 for model_type in sub_column_names2
    ]
    columns = pd.MultiIndex.from_tuples(column_tuples)
    # Define Multi-level rows
    row_names = MODEL_NAMES
    sub_row_names = ["wo_calibration", "w_calibration"] 
    row_tuples = [(model_name, cali) for model_name in row_names for cali in sub_row_names]
    rows= pd.MultiIndex.from_tuples(row_tuples)
    existing_df = pd.DataFrame(index=rows, columns=columns)
    existing_df.to_csv(result_file_path)

# Create log dir
full_log_path = os.path.join(primary_log_path, args.attack)
if os.path.exists(full_log_path) is False:
    print("Creating log directory:", full_log_path)
    os.makedirs(full_log_path)

# Configure logging
GlobalLogger.initialize(
    os.path.join(
        full_log_path, 
        f"{full_model_type}.log"
    )
)
logger = GlobalLogger.get_logger(__name__)
# Log initial configuration details
logger.info("Starting Attack")
logger.info(f"Arguments: {args}")
logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
# Load the dataset using the specified parameters
logger.info("Loading dataset...")
dataset = load_dataset(args.dataset, logger)
try:
    data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
    with open(data_split_file, 'rb') as file:
        data_split_info = pickle.load(file)
    logger.info("Data split information successfully loaded:")
except FileNotFoundError:
    logger.info(f"Error: The file '{data_split_file}' does not exist")
# Creating dataloader
train_idxs = data_split_info[0]["train"]
test_idxs = data_split_info[0]["test"]
logger.info(f"Dataset Specs : Train size {len(train_idxs)}, Test size {len(test_idxs)}")
logger.info("Creating dataloader...")
data_loader = get_dataloader_from_dataset(
    args.dataset, 
    Subset(dataset, np.concatenate((train_idxs, test_idxs), axis=0)), 
    batch_size=batch_size, 
    train=False
)
logger.info(f"Dataset loaded successfully. Batches: {len(data_loader)}")
# Load the specified model from the model pool
logger.info(f"Loading {full_model_type} model: {args.model} for dataset: {args.dataset}")
# Load trained models
target_model, reference_models = load_model(
    args.model, args.dataset, args.model_type, args.reference_models, primary_model_path, device, n_steps
)
# Perform attack and get results
attack = perform_MIA(
            attack_type = args.attack, 
            model_type = args.model_type,
            target_model = target_model, 
            reference_models = reference_models, 
            data_loader = data_loader, 
            device = device,
            n_steps = n_steps,
            calibration = args.calibration,
            dropout = args.dropout,
            n_samples = args.n_samples,
        )
attack.compute_scores()
results = attack.get_results()
logger.info(f"Results: \n {results}")
# Save Results
cali_type = "w_calibration" if args.calibration else "wo_calibration"
existing_df[(args.dataset, args.attack, half_model_type), (args.model, cali_type)] = results
existing_df.to_csv(result_file_path)