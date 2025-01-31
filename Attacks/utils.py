import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

def add_dimension(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

def compute_attack_results(mia_scores, target_memberships):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    fpr_list, tpr_list, thresholds = roc_curve(target_memberships.ravel(), mia_scores.ravel())
    roc_auc = auc(fpr_list, tpr_list)
    
    one_fpr = tpr_list[np.where(fpr_list <= 0.01)[0][-1]]
    one_tenth_fpr = tpr_list[np.where(fpr_list <= 0.001)[0][-1]]
    
    zero_fpr_indices = np.where(fpr_list == 0.0)[0]
    zero_fpr = tpr_list[zero_fpr_indices[-1]] if zero_fpr_indices.size > 0 else 0.0

    return {
        #"fpr": fpr_list,
        #"tpr": tpr_list,
        "auc": roc_auc,
        "one_fpr": one_fpr,
        "one_tenth_fpr": one_tenth_fpr,
        "zero_fpr": zero_fpr,
        #"thresholds": thresholds
    }

def visualization(results):
    """
    visualize the attack results (ROC curve)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["fpr"],
        results["tpr"],
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {results['auc']:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", lw=2, label="Random Guess")
    # Add grid, labels, and legend
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("TPR-FPR (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def generate_dropout_image(images, dropout_rate):
    """
    Create a function that applies dropout-style perturbation to input images.

    Args:
        dropout_rate (float): Probability of dropping a pixel (value set to 0).

    Returns:
        input_dropout_fn (callable): Function to apply dropout to images.
    """
    assert 0.0 <= dropout_rate <= 1.0, "Dropout rate must be between 0 and 1"
    mask = torch.rand_like(images, dtype=torch.float32) > dropout_rate
    return images * mask

def CamPoisson_fn(images, probability_matrix):
    """
    Augment images by retaining each pixel with a probability defined by a given matrix.

    Args:
        images (torch.Tensor): Input batch of images (shape: [batch_size, channels, height, width]).
        probability_matrix (torch.Tensor): Probability matrix (shape: [batch_size, height, width]),
                                            where each value defines the probability of retaining the pixel.

    Returns:
        augmented_images (torch.Tensor): Augmented images with pixels randomly retained or dropped.
    """
    # Ensure probability_matrix dimensions match the images batch size, height, and width
    assert images.shape[0] == probability_matrix.shape[0], \
        "Batch size of probability_matrix must match the batch size of images"
    assert images.shape[2:] == probability_matrix.shape[1:], \
        "Height and width of probability_matrix must match the height and width of images"
    # Expand probability_matrix to match the channel dimension of images
    probability_matrix = probability_matrix.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
    probability_matrix = probability_matrix.expand_as(images)  # Shape: [batch_size, channels, height, width]
    # Generate a mask using the probability matrix
    mask = torch.bernoulli(probability_matrix)  # Each pixel has probability p to be retained
    augmented_images = images * mask
    return augmented_images

def compute_confidence(model, data_loader, device, is_snn, n_steps, calibration, dropout_rate, n_samples):
    """
    Compute the confidence of each sample in the dataset.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The computation device (e.g., 'cpu' or 'cuda').
        is_snn (bool): Whether the model is an SNN.
        n_steps (int, optional): Number of time steps for SNN (required if is_snn is True).
        calibration (bool, optional): Whether to perform uncertainty-aware calibration.
        n_samples (int, optional): The number of augmented samples to generate per input if calibration is True.

    Returns:
        confidence_list (numpy.ndarray): The confidence of each sample (softmax probability for the correct class).
    """
    
    model.eval()
    model.to(device)
    confidence_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            if is_snn:
                images = add_dimension(images, n_steps)
            images = images.to(device)
            labels = labels.to(device)
            if calibration:
                augmented_confidences = torch.zeros((n_samples, images.size(0)), device=device)
                for i in range(n_samples):
                    augmented_images = generate_dropout_image(images, dropout_rate)
                    logits = model(augmented_images, L=0, t=n_steps).mean(1) if is_snn else model(augmented_images)
                    softmax_values = F.softmax(logits, dim=1)
                    batch_indices = torch.arange(labels.size(0), device=device)
                    confidences = softmax_values[batch_indices, labels]
                    augmented_confidences[i] = confidences
                confidences = augmented_confidences.mean(dim=0)
            else:
                logits = model(images, L=0, t=n_steps).mean(1) if is_snn else model(images)
                softmax_values = F.softmax(logits, dim=1)
                confidences = softmax_values[range(len(labels)), labels]
            confidence_list.append(confidences)
    confidence_list = torch.cat(confidence_list, dim=0).cpu().numpy()
    return confidence_list

