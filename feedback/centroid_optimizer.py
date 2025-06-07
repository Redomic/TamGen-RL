"""
Fixed centroid shift optimization for latent space molecular generation.
Addresses critical data flow issues and improves optimization strategy.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
import torch
import logging

from .reward_model import LatentRewardModel
from .reward_utils import compute_advanced_reward, analyze_reward_distribution


def centroid_shift_optimize(z_vectors: np.ndarray,
                          smiles_list: List[str],
                          docking_scores: List[Optional[float]],
                          latent_dim: int = 128,
                          lambda_sas: float = 0.3,
                          lambda_logp: float = 0.1,
                          lambda_mw: float = 0.1,
                          top_k: int = 50,
                          shift_alpha: float = 0.5,
                          noise_sigma: float = 0.05,
                          use_gradient_optimization: bool = True,
                          device: str = "auto",
                          reward_model_epochs: int = 100,
                          diversity_weight: float = 0.2) -> Tuple[List[np.ndarray], List[float], List[Dict[str, Any]]]:
    """
    FIXED: Compute new latent vectors using centroid shift based on reward.
    
    This version fixes the critical data flow bug where the reward model
    would clear its data before computing the centroid shift.
    
    Args:
        z_vectors: Array of latent vectors (shape: [n_samples, latent_dim])
        smiles_list: List of SMILES strings
        docking_scores: List of docking scores (can contain None values)
        latent_dim: Dimension of latent vectors
        lambda_sas: Weight for SAS penalty
        lambda_logp: Weight for LogP penalty  
        lambda_mw: Weight for MW penalty
        top_k: Number of best samples for centroid computation
        shift_alpha: Scaling factor for shift direction
        noise_sigma: Standard deviation for added noise
        use_gradient_optimization: Whether to use gradient-based optimization
        device: Device for reward model ("cpu", "cuda", or "auto")
        reward_model_epochs: Number of training epochs for reward model
        diversity_weight: Weight for molecular diversity bonus
        
    Returns:
        Tuple of (shifted_z_vectors, rewards, metrics_list)
    """
    
    # Input validation
    if not isinstance(z_vectors, np.ndarray):
        z_vectors = np.array(z_vectors)
    
    n_samples = len(z_vectors)
    assert n_samples == len(smiles_list) == len(docking_scores), \
        f"Input length mismatch: z_vectors={n_samples}, smiles={len(smiles_list)}, docking={len(docking_scores)}"
    
    if n_samples == 0:
        logging.warning("No samples provided for centroid shift optimization")
        return [], [], []
    
    # Validate latent dimension
    if z_vectors.shape[1] != latent_dim:
        logging.warning(f"Latent dimension mismatch: expected {latent_dim}, got {z_vectors.shape[1]}")
        latent_dim = z_vectors.shape[1]
    
    # Clamp top_k to available samples
    actual_top_k = min(top_k, n_samples)
    if actual_top_k < 5:
        logging.warning(f"Only {actual_top_k} samples available for centroid shift (recommended: ≥10)")
    
    print(f"🧠 Starting centroid shift optimization for {n_samples} samples")
    print(f"   Using top-{actual_top_k} samples for optimization")
    
    # Initialize reward model
    reward_model = LatentRewardModel(
        latent_dim=latent_dim,
        hidden_dim=min(256, latent_dim * 2),  # Adaptive hidden dimension
        device=device
    )
    
    # Step 1: Compute rewards for all molecules
    print("📊 Computing molecular rewards...")
    rewards = []
    metrics_list = []
    valid_molecules = []
    
    # Convert SMILES to molecules for diversity calculation
    reference_mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            reference_mols.append(mol)
    
    reward_weights = {
        'qed': 2.0,          # Maximize QED
        'sas': 1.5,           # Minimize SAS (converted to maximize)
        'lipinski': 1.0,      # Minimize violations
        'diversity': diversity_weight,  # Maximize diversity
        'docking': 3.0,       # Minimize docking score
        'logp': 1.5           # Keep in 0-5 range
    }
    
    for i, (z, smi, dock_score) in enumerate(zip(z_vectors, smiles_list, docking_scores)):
        mol = Chem.MolFromSmiles(smi)
        
        if mol is None:
            reward = -10.0
            metrics = {"error": "invalid_smiles", "smiles": smi}
            logging.warning(f"Invalid SMILES at index {i}: {smi}")
        else:
            # Compute advanced reward with diversity consideration
            reward, metrics = compute_advanced_reward(
                mol=mol,
                docking_score=dock_score,
                reference_mols=reference_mols[:i] if i > 0 else None,  # Previous molecules for diversity
                weights=reward_weights
            )
            valid_molecules.append(mol)
        
        # Add to reward model - CRITICAL: Do this BEFORE training
        reward_model.add(z, reward)
        rewards.append(reward)
        metrics_list.append(metrics)
    
    print(f"   ✓ Computed rewards for {len(valid_molecules)}/{n_samples} valid molecules")
    
    # Step 2: Train reward model BEFORE getting centroid shift
    print("🎯 Training reward model...")
    training_result = reward_model.train(epochs=reward_model_epochs, validation_split=0.2)
    
    if training_result["status"] != "success":
        logging.error(f"Reward model training failed: {training_result}")
        # Fallback to simple centroid shift without model
        return _fallback_centroid_shift(z_vectors, rewards, actual_top_k, shift_alpha, noise_sigma)
    
    print(f"   ✓ Training complete: {training_result['epochs_trained']} epochs, "
          f"final loss: {training_result.get('final_train_loss', 'N/A'):.4f}")
    
    # Step 3: Compute shift direction BEFORE clearing model data
    print("🧭 Computing optimal shift direction...")
    
    if use_gradient_optimization and len(valid_molecules) >= 10:
        # Use gradient-based optimization for better results
        direction = reward_model.get_gradient_based_shift(
            top_k=actual_top_k,
            lr=0.01,
            steps=50
        )
        optimization_method = "gradient-based"
    else:
        # Use simple centroid shift as fallback
        direction = reward_model.get_centroid_shift(
            top_k=actual_top_k,
            smiles_list=smiles_list,
            use_gradient=False
        )
        optimization_method = "centroid-based"
    
    # Validate shift direction
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        logging.warning("Shift direction is nearly zero, using random perturbation")
        direction = np.random.normal(0, 0.1, size=direction.shape)
        direction_norm = np.linalg.norm(direction)
        optimization_method += " (random fallback)"
    
    print(f"   ✓ {optimization_method} optimization complete, direction norm: {direction_norm:.4f}")
    
    # Step 4: Apply shift to all latent vectors
    print("🔄 Applying latent space shifts...")
    z_shifted_list = []
    
    # Adaptive noise based on shift magnitude
    adaptive_noise = min(noise_sigma, direction_norm * 0.1)
    
    for i, z in enumerate(z_vectors):
        # Apply shift with optional adaptive scaling
        base_shift = shift_alpha * direction
        
        # Add position-dependent noise for diversity
        noise = np.random.normal(0, adaptive_noise, size=z.shape)
        
        # Optional: Scale shift based on current reward (boost low-reward samples more)
        if len(rewards) > i:
            reward_percentile = np.percentile(rewards, 50)  # Median
            if rewards[i] < reward_percentile:
                boost_factor = 1.2  # Boost low-reward samples more
            else:
                boost_factor = 0.8  # Smaller shift for already good samples
            base_shift *= boost_factor
        
        shifted = np.array(z) + base_shift + noise
        z_shifted_list.append(shifted)
    
    # Step 5: Analyze results and provide feedback
    reward_analysis = analyze_reward_distribution(rewards, metrics_list)
    
    print("📈 Optimization summary:")
    print(f"   • Reward distribution: μ={reward_analysis['reward_stats']['mean']:.3f}, "
          f"σ={reward_analysis['reward_stats']['std']:.3f}")
    print(f"   • Valid molecules: {len(valid_molecules)}/{n_samples}")
    print(f"   • Shift magnitude: {direction_norm:.4f}")
    print(f"   • Applied noise: σ={adaptive_noise:.4f}")
    
    # Check diversity
    unique_smiles = len(set(smiles_list))
    diversity_ratio = unique_smiles / len(smiles_list)
    if diversity_ratio < 0.3:
        print(f"   ⚠️  Low diversity detected: {unique_smiles}/{len(smiles_list)} unique SMILES")
        print("      Consider increasing noise_sigma or reducing shift_alpha")
    else:
        print(f"   ✓ Good diversity: {unique_smiles}/{len(smiles_list)} unique SMILES")
    
    # Step 6: Clear model data to free memory (AFTER we're done with it)
    reward_model.clear_data()
    
    # Memory cleanup
    del reward_model, direction, reference_mols
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Centroid shift optimization complete: {len(z_shifted_list)} shifted vectors")
    
    return z_shifted_list, rewards, metrics_list


def _fallback_centroid_shift(z_vectors: np.ndarray, 
                           rewards: List[float],
                           top_k: int,
                           shift_alpha: float,
                           noise_sigma: float) -> Tuple[List[np.ndarray], List[float], List[Dict[str, Any]]]:
    """
    Fallback centroid shift when reward model training fails.
    
    Args:
        z_vectors: Array of latent vectors
        rewards: List of computed rewards
        top_k: Number of top samples to use
        shift_alpha: Shift scaling factor
        noise_sigma: Noise standard deviation
        
    Returns:
        Tuple of (shifted_vectors, rewards, metrics_list)
    """
    logging.warning("Using fallback centroid shift method")
    
    if len(rewards) == 0:
        return [], [], []
    
    # Get top-k indices
    top_indices = np.argsort(rewards)[-top_k:]
    top_z = z_vectors[top_indices]
    
    # Compute simple centroid shift
    top_centroid = np.mean(top_z, axis=0)
    all_centroid = np.mean(z_vectors, axis=0)
    direction = top_centroid - all_centroid
    
    # Apply shift
    z_shifted_list = []
    for z in z_vectors:
        noise = np.random.normal(0, noise_sigma, size=z.shape)
        shifted = z + shift_alpha * direction + noise
        z_shifted_list.append(shifted)
    
    # Create basic metrics
    metrics_list = [{"reward": r, "method": "fallback"} for r in rewards]
    
    return z_shifted_list, rewards, metrics_list


def adaptive_shift_schedule(iteration: int, 
                          max_iterations: int,
                          initial_alpha: float = 0.5,
                          final_alpha: float = 0.1,
                          schedule_type: str = "linear") -> float:
    """
    Compute adaptive shift alpha based on iteration number.
    
    Args:
        iteration: Current iteration (0-indexed)
        max_iterations: Total number of iterations
        initial_alpha: Starting shift magnitude
        final_alpha: Ending shift magnitude
        schedule_type: Type of schedule ("linear", "exponential", "cosine")
        
    Returns:
        Adaptive alpha value
    """
    if max_iterations <= 1:
        return initial_alpha
    
    progress = iteration / (max_iterations - 1)
    
    if schedule_type == "linear":
        alpha = initial_alpha + (final_alpha - initial_alpha) * progress
    elif schedule_type == "exponential":
        alpha = initial_alpha * (final_alpha / initial_alpha) ** progress
    elif schedule_type == "cosine":
        alpha = final_alpha + (initial_alpha - final_alpha) * 0.5 * (1 + np.cos(np.pi * progress))
    else:
        alpha = initial_alpha
    
    return float(alpha)


def multi_objective_centroid_shift(z_vectors: np.ndarray,
                                 smiles_list: List[str],
                                 objectives: Dict[str, List[float]],
                                 objective_weights: Dict[str, float],
                                 **kwargs) -> Tuple[List[np.ndarray], List[float], List[Dict[str, Any]]]:
    """
    Multi-objective centroid shift optimization.
    
    Args:
        z_vectors: Array of latent vectors
        smiles_list: List of SMILES strings
        objectives: Dictionary of objective names to value lists
        objective_weights: Dictionary of objective names to weights
        **kwargs: Additional arguments for centroid_shift_optimize
        
    Returns:
        Tuple of (shifted_vectors, combined_rewards, metrics_list)
    """
    # Combine objectives into single reward
    combined_rewards = []
    n_samples = len(z_vectors)
    
    for i in range(n_samples):
        combined_reward = 0
        for obj_name, obj_values in objectives.items():
            if i < len(obj_values):
                weight = objective_weights.get(obj_name, 1.0)
                combined_reward += weight * obj_values[i]
        combined_rewards.append(combined_reward)
    
    # Use combined rewards as docking scores
    return centroid_shift_optimize(
        z_vectors=z_vectors,
        smiles_list=smiles_list,
        docking_scores=combined_rewards,
        **kwargs
    )