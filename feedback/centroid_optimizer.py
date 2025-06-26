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
    Ultra-simple optimization for cases with very few samples (3-10).
    Just shifts towards the best performing sample.
    
    Args:
        z_vectors: Array of latent vectors
        smiles_list: List of SMILES strings  
        rewards: List of computed rewards
        shift_alpha: Scaling factor for shift direction
        noise_sigma: Standard deviation for added noise
        
    Returns:
        Tuple of (shifted_z_vectors, rewards, metrics_list) - same format as centroid_shift_optimize
    """
    if len(rewards) < 2:
        # No optimization possible, just add noise
        z_shifted_list = [z + np.random.normal(0, noise_sigma, size=z.shape) for z in z_vectors]
        print(f"⚡ Ultra-simple optimization: added noise to {len(z_vectors)} samples (insufficient data for optimization)")
        return np.stack(z_shifted_list), rewards, []
    
    # Find best sample
    best_idx = np.argmax(rewards)
    best_z = z_vectors[best_idx]
    best_reward = rewards[best_idx]
    
    print(f"🧭 Computing simple shift directions...")
    print(f"   🏆 Best sample (idx={best_idx}): reward={best_reward:.3f}")
    
    # Shift all samples towards best
    z_shifted_list = []
    direction_norms = []
    
    for i, z in enumerate(z_vectors):
        direction = best_z - z
        direction_norm = np.linalg.norm(direction)
        direction_norms.append(direction_norm)
        
        noise = np.random.normal(0, noise_sigma, size=z.shape)
        shifted = z + shift_alpha * direction + noise
        z_shifted_list.append(shifted)
    
    # Compute statistics
    avg_direction_norm = np.mean(direction_norms)
    max_direction_norm = np.max(direction_norms)
    
    print(f"   ✓ Simple shift directions computed:")
    print(f"     • Average direction norm: {avg_direction_norm:.4f}")
    print(f"     • Max direction norm: {max_direction_norm:.4f}")
    print(f"     • Shift magnitude (α={shift_alpha}): {shift_alpha * avg_direction_norm:.4f}")
    print(f"     • Applied noise: σ={noise_sigma:.4f}")
    
    print(f"⚡ Ultra-simple optimization: shifted {len(z_vectors)} samples towards best")
    
    # Create simple metrics (empty list since we're not recomputing rewards)
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
                          weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
    """
    Simplified centroid shift optimization for 3 criteria: QED, SAS, Binding Affinity.
    
    NO FALLBACKS - ALL CRITERIA MUST SUCCEED OR THE FUNCTION WILL RAISE AN EXCEPTION.
    
    Args:
        z_vectors: Array of latent vectors (shape: [n_samples, latent_dim])
        smiles_list: List of SMILES strings
        docking_scores: List of docking scores (can contain None values)
        latent_dim: Dimension of latent vectors
        top_k: Number of best samples for centroid computation
        shift_alpha: Scaling factor for shift direction
        noise_sigma: Standard deviation for added noise
        device: Device for reward model ("cpu", "cuda", or "auto")
        epochs: Number of training epochs for reward model
        pdb_id: PDB ID for GNINA docking (e.g., "1HSG" for HIV protease)
        use_binding_affinity: Whether to compute binding affinity using GNINA
        weights: Optional weights for the three criteria
        
    Returns:
        Tuple of (shifted_z_vectors, rewards, metrics_list)
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If reward calculation or model training fails
    """
    
    # Input validation
    if not isinstance(z_vectors, np.ndarray):
        z_vectors = np.array(z_vectors)
    
    n_samples = len(z_vectors)
    if n_samples != len(smiles_list) or n_samples != len(docking_scores):
        raise ValueError(f"Input length mismatch: z_vectors={n_samples}, smiles={len(smiles_list)}, docking={len(docking_scores)}")
    
    if n_samples == 0:
        raise ValueError("No samples provided for centroid shift optimization")
    
    # Validate latent dimension
    if z_vectors.shape[1] != latent_dim:
        logging.warning(f"Latent dimension mismatch: expected {latent_dim}, got {z_vectors.shape[1]}")
        latent_dim = z_vectors.shape[1]
    
    # Clamp top_k to available samples
    actual_top_k = min(top_k, n_samples)
    
    print(f"🧠 Starting simplified centroid shift optimization for {n_samples} samples")
    print(f"   Using top-{actual_top_k} samples for optimization")
    
    # Use ultra-simple optimization for very few samples (3-10)
    # or when we have fewer samples than required for reward model training
    min_samples_for_training = 11  # Same as LatentRewardModel requirement
    if n_samples <= 10 or n_samples < min_samples_for_training:
        print(f"⚡ Using ultra-simple optimization for {n_samples} samples (too few for full optimization)")
        
        # First compute rewards for all molecules
        print("📊 Computing molecular rewards with 3 criteria...")
        rewards = []
        metrics_list = []
        
        # Set up default weights with binding affinity prioritized
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
                    use_binding_affinity=use_binding_affinity
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compute reward for molecule {i} (SMILES: {smi}): {e}")
            
            rewards.append(reward)
            metrics_list.append(metrics)
        
        print(f"   ✓ Computed rewards for {n_samples} molecules")
        
        # Show binding affinity statistics if available
        binding_affinities = [m.get('binding_affinity') for m in metrics_list if m.get('binding_affinity') is not None]
        if binding_affinities:
            best_affinity = min(binding_affinities)
            success_rate = len(binding_affinities) / n_samples
            print(f"   🎯 Binding affinity: {len(binding_affinities)}/{n_samples} successful ({success_rate:.1%})")
            print(f"   🏆 Best binding affinity: {best_affinity:.2f} kcal/mol")
        
        # Use simple optimization
        z_shifted_array, _, _ = simple_optimizer(
            z_vectors=z_vectors,
            smiles_list=smiles_list,
            rewards=rewards,
            shift_alpha=shift_alpha,
            noise_sigma=noise_sigma
        )
        
        print(f"✅ Ultra-simple optimization complete: {len(z_shifted_array)} shifted vectors")
        return z_shifted_array, rewards, metrics_list
    
    # Initialize reward model
    reward_model = LatentRewardModel(
        latent_dim=latent_dim,
        hidden_dim=min(256, latent_dim * 2),
        device=device
    )
    
    # Step 1: Compute rewards for all molecules
    print("📊 Computing molecular rewards with 3 criteria...")
    rewards = []
    metrics_list = []
    
    # Set up default weights with binding affinity prioritized
    if weights is None:
        weights = {
            'qed': 1.0,
            'sas': 1.0,
            'binding_affinity': 3.0  # Higher weight for binding affinity
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
        
        # Compute 3-criteria reward - will raise exception on failure
        try:
            reward, metrics = compute_three_criterion_reward(
                mol=mol,
                docking_score=dock_score,
                pdb_id=pdb_id if use_binding_affinity else None,
                weights=weights,
                use_binding_affinity=use_binding_affinity
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute reward for molecule {i} (SMILES: {smi}): {e}")
        
        # Add to reward model
        reward_model.add(z, reward)
        rewards.append(reward)
        metrics_list.append(metrics)
    
    print(f"   ✓ Computed rewards for {n_samples} molecules")
    
    # Show binding affinity statistics if available
    binding_affinities = [m.get('binding_affinity') for m in metrics_list if m.get('binding_affinity') is not None]
    if binding_affinities:
        best_affinity = min(binding_affinities)  # Most negative = best
        success_rate = len(binding_affinities) / n_samples
        print(f"   🎯 Binding affinity: {len(binding_affinities)}/{n_samples} successful ({success_rate:.1%})")
        print(f"   🏆 Best binding affinity: {best_affinity:.2f} kcal/mol")
    
    # Step 2: Train reward model
    print("🎯 Training reward model...")
    training_result = reward_model.train(epochs=epochs)
    
    if training_result["status"] != "success":
        raise RuntimeError(f"Reward model training failed: {training_result}")
    
    print(f"   ✓ Training complete: {training_result['epochs_trained']} epochs, "
          f"final loss: {training_result.get('final_loss', 'N/A'):.4f}")
    
    # Step 3: Compute centroid shift direction
    print("🧭 Computing centroid shift direction...")
    direction = reward_model.get_centroid_shift(top_k=actual_top_k)
    
    # Validate shift direction
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise RuntimeError("Shift direction is nearly zero - optimization failed")
    
    print(f"   ✓ Centroid shift computed, direction norm: {direction_norm:.4f}")
    
    # Step 4: Apply shift to all latent vectors
    print("🔄 Applying latent space shifts...")
    z_shifted_list = []
    
    for z in z_vectors:
        # Apply shift with noise
        noise = np.random.normal(0, noise_sigma, size=z.shape)
        shifted = np.array(z) + shift_alpha * direction + noise
        z_shifted_list.append(shifted)
    
    # Step 5: Analyze results
    try:
        reward_analysis = analyze_reward_distribution(rewards, metrics_list)
    except Exception as e:
        raise RuntimeError(f"Failed to analyze reward distribution: {e}")
    
    print("📈 Optimization summary:")
    print(f"   • Reward distribution: μ={reward_analysis['reward_stats']['mean']:.3f}, "
          f"σ={reward_analysis['reward_stats']['std']:.3f}")
    print(f"   • All molecules processed successfully: {n_samples}/{n_samples}")
    
    # Show binding affinity summary
    if 'binding_affinity_stats' in reward_analysis:
        ba_stats = reward_analysis['binding_affinity_stats']
        print(f"   • Binding affinity: μ={ba_stats['mean']:.2f}, best={ba_stats['best_binding']:.2f} kcal/mol")
        print(f"   • Docking success rate: {reward_analysis['binding_affinity_success_rate']:.1%}")
    
    # Show criteria availability
    if 'criteria_availability' in reward_analysis:
        avail = reward_analysis['criteria_availability']
        print(f"   • Criteria availability: QED={avail.get('qed', 0):.1%}, "
              f"SAS={avail.get('sas', 0):.1%}, Binding={avail.get('binding_affinity', 0):.1%}")
    
    print(f"   • Shift magnitude: {direction_norm:.4f}")
    print(f"   • Applied noise: σ={noise_sigma:.4f}")
    
    # Step 6: Clean up
    reward_model.clear_data()
    del reward_model
    
    print(f"✅ Centroid shift optimization complete: {len(z_shifted_list)} shifted vectors")
    
    # Convert list to numpy array for consistency
    z_shifted_array = np.stack(z_shifted_list)
    
    return z_shifted_array, rewards, metrics_list