import numpy as np
from rdkit import Chem
from feedback.reward_utils import compute_reward
from feedback.reward_model import LatentRewardModel

def centroid_shift_optimize(z_vectors, smiles_list, docking_scores, latent_dim=128,
                             lambda_sas=0.3, lambda_logp=0.1, lambda_mw=0.1,
                             top_k=50, shift_alpha=0.5):
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

    reward_model = LatentRewardModel(latent_dim)
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
    direction = reward_model.get_centroid_shift(top_k=top_k)

    z_shifted_list = [np.array(z) + shift_alpha * direction for z in z_vectors]

    return z_shifted_list, rewards, metrics_list