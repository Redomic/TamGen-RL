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

    # Clamp top_k to available latents/SMILES and warn if clamped
    top_k = min(top_k, len(z_vectors))
    if top_k < 5:
        print(f"[WARNING] Only {top_k} valid molecules available for centroid shift (reduce top_k or increase diversity).")

    z_vectors = np.array(z_vectors)

    reward_model = LatentRewardModel(latent_dim)
    reward_model
    rewards = []
    metrics_list = []

    for z, smi, dock in zip(z_vectors, smiles_list, docking_scores):
        mol = Chem.MolFromSmiles(smi)
        reward, metrics = compute_reward(mol, docking_score=dock,
                                         lambda_sas=lambda_sas,
                                         lambda_logp=lambda_logp,
                                         lambda_mw=lambda_mw)
        reward_model.add(z, reward)
        rewards.append(reward)
        metrics_list.append(metrics)

    reward_model.train()
    # Print devices of underlying torch model parameters for debugging
    print("Model parameters devices:", [p.device for p in reward_model.model.parameters()])
    # Print devices of stored tensors in z_list, if elements are tensors; otherwise, indicate 'cpu'
    # (z_list may contain numpy arrays or torch tensors depending on implementation)
    stored_devices = []
    for t in reward_model.z_list:
        # Check if t is a torch tensor with .device attribute
        if hasattr(t, "device"):
            stored_devices.append(t.device)
        else:
            # Assume numpy arrays or other, which reside on CPU
            stored_devices.append("cpu")
    print("Stored tensors devices:", stored_devices)
    direction = reward_model.get_centroid_shift(top_k=top_k, smiles_list=smiles_list)

    reward_model.z_list.clear()
    reward_model.r_list.clear()

    z_shifted_list = [
        np.array(z) + shift_alpha * direction + np.random.normal(0, noise_sigma, size=z.shape)
        for z in z_vectors
    ]

    # Log diversity stats
    print(f"Unique SMILES this iteration: {len(set(smiles_list))} / {len(smiles_list)}")

    result = (z_shifted_list, rewards, metrics_list)

    # Explicitly delete large temporary variables and clear CUDA cache to help avoid OOM

    del reward_model, rewards, metrics_list, z_vectors, direction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result