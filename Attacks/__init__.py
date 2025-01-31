from .main import *

def get_attack_instance(attack_type, target_model, reference_models, data_loader, device, args):
    attack_classes = {
        "attack_p": Attack_P,
        "attack_r": Attack_R,
        "rmia": Attack_RMIA
    }
    
    if attack_type not in attack_classes:
        raise ValueError(f"Invalid attack type: {attack_type}")
    
    attack_class = attack_classes[attack_type]
    attack_instance = attack_class(
        target_model, data_loader, device, args.model_type, args.t, 
        args.calibration, args.dropout, args.n_samples, reference_models
    )
    return attack_instance