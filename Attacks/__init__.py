from .main import *

def perform_MIA(
        attack_type, 
        model_type,
        target_model, 
        reference_models, 
        data_loader, 
        device,
        n_steps,
        calibration,
        dropout,
        n_samples,
    ):
    attack_classes = {
        "attack_p": Attack_P,
        "attack_r": Attack_R,
        "rmia": Attack_RMIA
    }
    
    if attack_type not in attack_classes:
        raise ValueError(f"Invalid attack type: {attack_type}")
    
    attack_class = attack_classes[attack_type]
    attack_instance = attack_class(
        target_model = target_model,
        reference_models =  reference_models,
        data_loader = data_loader, 
        device = device, 
        model_type = model_type, 
        n_steps = n_steps, 
        calibration = calibration, 
        dropout = dropout, 
        n_samples = n_samples, 
    )
    return attack_instance