import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
import logging

from .reward_model import LatentRewardModel
from .reward_utils import compute_three_criterion_reward, analyze_reward_distribution


def simple_optimizer(z_vectors: np.ndarray,
                   smiles_list: List[str],
                   rewards: List[float],
                   shift_alpha: float = 0.3,
                   noise_sigma: float = 0.05) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
    """
    A simplified optimization strategy for small sample sizes, shifting all latent 
    vectors toward the single best-performing sample.
    
    Args:
        z_vectors: An array of latent vectors.
        smiles_list: A list of SMILES strings corresponding to the latent vectors.
        rewards: A list of rewards for each molecule.
        shift_alpha: The scaling factor for the shift direction.
        noise_sigma: The standard deviation of the Gaussian noise to add.
        
    Returns:
        A tuple containing the shifted latent vectors, original rewards, and an
        empty list for metrics (as none are computed).
    """
    if len(rewards) < 2:
        z_shifted_list = [z + np.random.normal(0, noise_sigma, size=z.shape) for z in z_vectors]
        print(f"⚡ Ultra-simple optimization: added noise to {len(z_vectors)} samples (insufficient data for optimization)")
        return np.stack(z_shifted_list), rewards, []
    
    best_idx = np.argmax(rewards)
    best_z = z_vectors[best_idx]
    best_reward = rewards[best_idx]
    
    print(f"🧭 Computing simple shift directions...")
    print(f"   🏆 Best sample (idx={best_idx}): reward={best_reward:.3f}")
    
    z_shifted_list = []
    direction_norms = []
    
    for i, z in enumerate(z_vectors):
        direction = best_z - z
        direction_norm = np.linalg.norm(direction)
        direction_norms.append(direction_norm)
        
        noise = np.random.normal(0, noise_sigma, size=z.shape)
        shifted = z + shift_alpha * direction + noise
        z_shifted_list.append(shifted)
    
    avg_direction_norm = np.mean(direction_norms)
    max_direction_norm = np.max(direction_norms)
    
    print(f"   ✓ Simple shift directions computed:")
    print(f"     • Average direction norm: {avg_direction_norm:.4f}")
    print(f"     • Max direction norm: {max_direction_norm:.4f}")
    print(f"     • Shift magnitude (α={shift_alpha}): {shift_alpha * avg_direction_norm:.4f}")
    print(f"     • Applied noise: σ={noise_sigma:.4f}")
    
    print(f"⚡ Ultra-simple optimization: shifted {len(z_vectors)} samples towards best")
    
    metrics_list = []
    
    return np.stack(z_shifted_list), rewards, metrics_list


def centroid_shift_optimize(z_vectors: np.ndarray,
                          smiles_list: List[str],
                          docking_scores: List[Optional[float]],
                          latent_dim: int = 128,
                          top_k: int = 50,
                          shift_alpha: float = 0.5,
                          noise_sigma: float = 0.05,
                          device: str = "auto",
                          epochs: int = 50,
                          pdb_id: Optional[str] = None,
                          use_binding_affinity: bool = True,
                          weights: Optional[Dict[str, float]] = None,
                          affinity_config: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
    """
    Optimizes latent vectors by shifting them towards a centroid of high-reward samples.
    
    This implementation relies on three criteria (QED, SAS, and Binding Affinity) and
    is designed to be strict; it will raise an exception if any of the underlying
    reward calculations or model training steps fail.
    
    Args:
        z_vectors: An array of latent vectors.
        smiles_list: A list of SMILES strings.
        docking_scores: A list of pre-computed docking scores (may contain None).
        latent_dim: The dimensionality of the latent vectors.
        top_k: The number of best-performing samples to use for centroid calculation.
        shift_alpha: The scaling factor for the shift direction.
        noise_sigma: The standard deviation of the Gaussian noise to add.
        device: The device for the reward model ('cpu', 'cuda', or 'auto').
        epochs: The number of epochs for training the reward model.
        pdb_id: The PDB ID for GNINA docking.
        use_binding_affinity: If True, enables binding affinity as an optimization criterion.
        weights: Optional dictionary of weights for the reward criteria.
        affinity_config: Optional dictionary of parameters for the binding affinity scaling.
        
    Returns:
        A tuple containing the shifted latent vectors, the computed rewards for the
        original vectors, and a list of detailed metrics for each molecule.
    """
    if not isinstance(z_vectors, np.ndarray):
        z_vectors = np.array(z_vectors)
    
    n_samples = len(z_vectors)
    if n_samples != len(smiles_list) or n_samples != len(docking_scores):
        raise ValueError(f"Input length mismatch: z_vectors={n_samples}, smiles={len(smiles_list)}, docking={len(docking_scores)}")
    
    if n_samples == 0:
        raise ValueError("No samples provided for centroid shift optimization")
    
    if z_vectors.shape[1] != latent_dim:
        logging.warning(f"Latent dimension mismatch: expected {latent_dim}, got {z_vectors.shape[1]}")
        latent_dim = z_vectors.shape[1]
    
    actual_top_k = min(top_k, n_samples)
    
    print(f"🧠 Starting simplified centroid shift optimization for {n_samples} samples")
    print(f"   Using top-{actual_top_k} samples for optimization")
    
    min_samples_for_training = 11
    if n_samples <= 10 or n_samples < min_samples_for_training:
        print(f"⚡ Using ultra-simple optimization for {n_samples} samples (too few for full optimization)")
        
        print("📊 Computing molecular rewards with 3 criteria...")
        rewards = []
        metrics_list = []
        
        if weights is None:
            weights = {
                'qed': 1.0,
                'sas': 1.0,
                'binding_affinity': 3.0
            }
        
        for i, (z, smi, dock_score) in enumerate(zip(z_vectors, smiles_list, docking_scores)):
            mol = Chem.MolFromSmiles(smi)
            
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {i}: {smi}")
            
            try:
                reward, metrics = compute_three_criterion_reward(
                    mol=mol,
                    docking_score=dock_score,
                    pdb_id=pdb_id if use_binding_affinity else None,
                    weights=weights,
                    use_binding_affinity=use_binding_affinity,
                    affinity_config=affinity_config
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compute reward for molecule {i} (SMILES: {smi}): {e}")
            
            rewards.append(reward)
            metrics_list.append(metrics)
        
        print(f"   ✓ Computed rewards for {n_samples} molecules")
        
        binding_affinities = [m.get('binding_affinity') for m in metrics_list if m.get('binding_affinity') is not None]
        if binding_affinities:
            best_affinity = min(binding_affinities)
            success_rate = len(binding_affinities) / n_samples
            print(f"   🎯 Binding affinity: {len(binding_affinities)}/{n_samples} successful ({success_rate:.1%})")
            print(f"   🏆 Best binding affinity: {best_affinity:.2f} kcal/mol")
        
        z_shifted_array, _, _ = simple_optimizer(
            z_vectors=z_vectors,
            smiles_list=smiles_list,
            rewards=rewards,
            shift_alpha=shift_alpha,
            noise_sigma=noise_sigma
        )
        
        print(f"✅ Ultra-simple optimization complete: {len(z_shifted_array)} shifted vectors")
        return z_shifted_array, rewards, metrics_list
    
    reward_model = LatentRewardModel(
        latent_dim=latent_dim,
        hidden_dim=min(256, latent_dim * 2),
        device=device
    )
    
    print("📊 Computing molecular rewards with 3 criteria...")
    rewards = []
    metrics_list = []
    
    if weights is None:
        weights = {
            'qed': 1.0,
            'sas': 1.0,
            'binding_affinity': 3.0
        }
    
    if use_binding_affinity and pdb_id:
        print(f"   🧬 Using GNINA docking with PDB ID: {pdb_id}")
    elif use_binding_affinity:
        print("   📋 Using pre-computed docking scores")
    else:
        print("   ⚗️  Using QED and SAS only")
    
    for i, (z, smi, dock_score) in enumerate(zip(z_vectors, smiles_list, docking_scores)):
        mol = Chem.MolFromSmiles(smi)
        
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        
        try:
            reward, metrics = compute_three_criterion_reward(
                mol=mol,
                docking_score=dock_score,
                pdb_id=pdb_id if use_binding_affinity else None,
                weights=weights,
                use_binding_affinity=use_binding_affinity,
                affinity_config=affinity_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute reward for molecule {i} (SMILES: {smi}): {e}")
        
        reward_model.add(z, reward)
        rewards.append(reward)
        metrics_list.append(metrics)
    
    print(f"   ✓ Computed rewards for {n_samples} molecules")
    
    binding_affinities = [m.get('binding_affinity') for m in metrics_list if m.get('binding_affinity') is not None]
    if binding_affinities:
        best_affinity = min(binding_affinities)
        success_rate = len(binding_affinities) / n_samples
        print(f"   🎯 Binding affinity: {len(binding_affinities)}/{n_samples} successful ({success_rate:.1%})")
        print(f"   🏆 Best binding affinity: {best_affinity:.2f} kcal/mol")
    
    print("🎯 Training reward model...")
    training_result = reward_model.train(epochs=epochs)
    
    if training_result["status"] != "success":
        raise RuntimeError(f"Reward model training failed: {training_result}")
    
    print(f"   ✓ Training complete: {training_result['epochs_trained']} epochs, "
          f"final loss: {training_result.get('final_loss', 'N/A'):.4f}")
    
    print("🧭 Computing centroid shift direction...")
    direction = reward_model.get_centroid_shift(top_k=actual_top_k)
    
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise RuntimeError("Shift direction is nearly zero - optimization failed")
    
    print(f"   ✓ Centroid shift computed, direction norm: {direction_norm:.4f}")
    
    print("🔄 Applying latent space shifts...")
    z_shifted_list = []
    
    for z in z_vectors:
        noise = np.random.normal(0, noise_sigma, size=z.shape)
        shifted = np.array(z) + shift_alpha * direction + noise
        z_shifted_list.append(shifted)
    
    try:
        reward_analysis = analyze_reward_distribution(rewards, metrics_list)
    except Exception as e:
        raise RuntimeError(f"Failed to analyze reward distribution: {e}")
    
    print("📈 Optimization summary:")
    print(f"   • Reward distribution: μ={reward_analysis['reward_stats']['mean']:.3f}, "
          f"σ={reward_analysis['reward_stats']['std']:.3f}")
    print(f"   • All molecules processed successfully: {n_samples}/{n_samples}")
    
    if 'binding_affinity_stats' in reward_analysis:
        ba_stats = reward_analysis['binding_affinity_stats']
        print(f"   • Binding affinity: μ={ba_stats['mean']:.2f}, best={ba_stats['best_binding']:.2f} kcal/mol")
        print(f"   • Docking success rate: {reward_analysis['binding_affinity_success_rate']:.1%}")
    
    if 'criteria_availability' in reward_analysis:
        avail = reward_analysis['criteria_availability']
        print(f"   • Criteria availability: QED={avail.get('qed', 0):.1%}, "
              f"SAS={avail.get('sas', 0):.1%}, Binding={avail.get('binding_affinity', 0):.1%}")
    
    print(f"   • Shift magnitude: {direction_norm:.4f}")
    print(f"   • Applied noise: σ={noise_sigma:.4f}")
    
    reward_model.clear_data()
    del reward_model
    
    print(f"✅ Centroid shift optimization complete: {len(z_shifted_list)} shifted vectors")
    
    z_shifted_array = np.stack(z_shifted_list)
    
    return z_shifted_array, rewards, metrics_list