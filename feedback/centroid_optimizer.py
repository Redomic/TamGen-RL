import numpy as np
from rdkit import Chem
import torch

from feedback.reward_utils import compute_reward
from feedback.reward_model import LatentRewardModel

def centroid_shift_optimize(z_vectors, smiles_list, docking_scores, latent_dim=128,
                             lambda_sas=0.3, lambda_logp=0.1, lambda_mw=0.1,
                             top_k=50, shift_alpha=0.5, noise_sigma=0.05):
    """
    Compute new latent vectors using centroid shift based on reward.

    Args:
        z_vectors: List of latent vectors (np arrays)
        smiles_list: List of SMILES strings
        docking_scores: List of float docking scores
        lambda_sas, lambda_logp, lambda_mw: reward hyperparameters
        top_k: number of best samples to compute centroid shift
        shift_alpha: scaling factor for shift

    Returns:
        z_shifted_list: list of shifted z vectors
        rewards: list of scalar reward values
        metrics_list: list of per-sample metric dicts
    """
    assert len(z_vectors) == len(smiles_list) == len(docking_scores)

    # Convert to numpy array early and check
    z_vectors = np.array(z_vectors)

    # Clamp top_k to available latents/SMILES and warn if clamped
    actual_top_k = min(top_k, len(z_vectors))
    if actual_top_k < 5:
        print(f"[WARNING] Only {actual_top_k} valid molecules available for centroid shift (reduce top_k or increase diversity).")

    reward_model = LatentRewardModel(latent_dim)
    rewards = []
    metrics_list = []

    for i, (z, smi, dock) in enumerate(zip(z_vectors, smiles_list, docking_scores)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reward, metrics = -10.0, {"error": "invalid_smiles"}
        else:
            reward, metrics = compute_reward(mol, docking_score=dock,
                                             lambda_sas=lambda_sas,
                                             lambda_logp=lambda_logp,
                                             lambda_mw=lambda_mw)
        
        reward_model.add(z, reward)
        rewards.append(reward)
        metrics_list.append(metrics)

    # Train the reward model
    reward_model.train()
    
    # Check if the reward model was cleared (this is the bug!)
    if len(reward_model.z_list) == 0:
        # Re-add the data since it was cleared
        for z, reward in zip(z_vectors, rewards):
            reward_model.add(z, reward)
    
    # Get centroid shift
    direction = reward_model.get_centroid_shift(top_k=actual_top_k, smiles_list=smiles_list)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-8:
        print(f"WARNING - Direction vector is nearly zero! Using random direction.")
        direction = np.random.normal(0, 0.1, size=direction.shape)

    # Apply centroid shift
    z_shifted_list = []
    for i, z in enumerate(z_vectors):
        shifted = np.array(z) + shift_alpha * direction + np.random.normal(0, noise_sigma, size=z.shape)
        z_shifted_list.append(shifted)

    # Log diversity stats
    unique_count = len(set(smiles_list))
    total_count = len(smiles_list)
    
    if unique_count < total_count * 0.1:  # Less than 10% unique
        print(f"WARNING - Very low diversity! Consider increasing noise or reducing shift.")

    # Don't clear the reward model lists here - let them clear naturally
    result = (z_shifted_list, rewards, metrics_list)

    # Clean up
    del reward_model, direction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result