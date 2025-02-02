import numpy as np
from torch import nn
import torch
from .utils import *
import warnings
warnings.filterwarnings("ignore")
from utils import GlobalLogger


TARGET_MEMBERSHIPS = np.array([1] * 30000 + [0] * 30000)
logger = GlobalLogger.get_logger(__name__)

class BaseAttack:
    def __init__(
        self, target_model, data_loader, device, model_type="ann", 
        n_steps=None, calibration=False, dropout=None, n_samples=None, reference_models=None
    ):
        """
        Base class for different attack types.
        
        Args:
            target_model: The target model (ANN or SNN).
            data_loader: The dataset to use.
            device: The device (CPU/GPU).
            model_type: Type of model ("ann" or "snn").
            n_steps: (Optional) Number of steps for SNN models.
            calibration: (Optional) If calibration needs to be done.
            dropout: (Optional) Dropout value for calibration.
            n_samples: (Optional) Number of augmentation samples for calibration attack.
            reference_models: (Optional) List of reference models (for Attack_R & Attack_RMIA).
        """
        self.target_model = target_model
        self.data_loader = data_loader
        self.device = device
        self.n_steps = n_steps
        self.calibration = calibration
        self.dropout = dropout
        self.n_samples = n_samples
        self.reference_models = reference_models or []
        self.model_type = model_type 

        # if model_type == "snn":
        #     assert n_steps is not None, "n_steps must be provided for SNN models."
        #     self.is_snn = True
        # elif model_type == "ann":
        #     self.is_snn = False
        # else:
        #     logger.info(f"Incorrect model type: {model_type}")

        if self.calibration:
            assert dropout is not None, "dropout must be provided for calibration"
            assert n_samples is not None, "n_samples must be provided for calibration"

    def compute_base_confidence(self, model, model_type_val):
        """Helper function to compute confidence scores."""
        logger.info("Computing confidence scores ...")
        if model_type_val=="snn":
            assert self.n_steps is not None, "n_steps must be provided for SNN models."
            is_snn=True
        elif model_type_val=="ann":
            is_snn=False
        else:
            logger.info(f"Incorrect model type: {model_type_val}")
        scores =  compute_confidence(
            model, self.data_loader, self.device, is_snn, 
            self.n_steps, self.calibration, self.dropout, self.n_samples
        )
        return scores

    def get_results(self):
        logger.info("Getting results ...")
        self.results = compute_attack_results(self.scores, TARGET_MEMBERSHIPS)


class Attack_P(BaseAttack):
    def compute_scores(self):
        """Computes the Attack-P scores for the target model."""
        self.scores = self.compute_base_confidence(self.target_model, self.model_type[0])


class Attack_R(BaseAttack):
    def compute_scores(self):
        """Computes the Attack-R scores for the target model."""
        self.target_model.eval()
        target_confidences = np.expand_dims(
            self.compute_base_confidence(self.target_model, self.model_type[0]), 
            axis=0
        )
        reference_confidences = []
        for i, ref_model in enumerate(self.reference_models):
            ref_model.eval()
            reference_confidences.append(
                self.compute_base_confidence(ref_model, self.model_type[i])
            )
        
        reference_confidences = np.array(reference_confidences)
        # Compute ratio and attack scores
        ratios = target_confidences / reference_confidences
        satisfied_conditions = (ratios >= 1).astype(np.float32)
        self.scores = np.mean(satisfied_conditions, axis=0)


class Attack_RMIA(Attack_R):
    def compute_scores(self):
        """Computes the Attack-RMIA scores for the target model."""
        self.target_model.eval()
        target_confidences = np.expand_dims(
            self.compute_base_confidence(self.target_model, self.model_type[0]), 
            axis=0
        )
        reference_confidences = []
        for i, ref_model in enumerate(self.reference_models):
            ref_model.eval()
            reference_confidences.append(
                self.compute_base_confidence(ref_model, self.model_type[i])
            )
        reference_confidences = np.array(reference_confidences)
        pr_x = np.mean(reference_confidences, axis=0)
        # Compute RMIA scores
        self.scores = target_confidences / pr_x
