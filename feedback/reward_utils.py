import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import GNINA docking function
from fairseq.molecule_utils.basic.run_gnina_docking import docking

# Import existing scoring functions from mol_scores.py
from fairseq.molecule_utils.basic.mol_scores import qed_score, calculate_sa_score
from rdkit import Chem


def compute_binding_affinity(pdb_id: str, smiles: str, max_retries: int = 3) -> float:
    """Compute binding affinity using GNINA docking. Raises exception on failure."""
    if not smiles or not pdb_id:
        raise ValueError(f"Invalid input: pdb_id='{pdb_id}', smiles='{smiles}'")
    
    # Validate SMILES first
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    for attempt in range(max_retries):
        try:
            affinity = docking(pdb_id=pdb_id, ligand_smiles=smiles)
            
            # Validate the result
            if affinity is not None and isinstance(affinity, (int, float)):
                # Reasonable range check for binding affinity (-20 to +5 kcal/mol)
                if -20.0 <= affinity <= 5.0:
                    return float(affinity)
            
            raise RuntimeError(f"Invalid binding affinity result: {affinity}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Docking attempt {attempt + 1} failed for {smiles[:20]}...: {e}")
                continue
            else:
                raise RuntimeError(f"All {max_retries} docking attempts failed for {smiles}: {e}")


def compute_qed(mol: Chem.Mol) -> float:
    """Compute QED (Quantitative Estimate of Drug-likeness) score. Raises exception on failure."""
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    try:
        qed_result = qed_score(mol)  # Use existing function from mol_scores.py
        if qed_result is None or not isinstance(qed_result, (int, float)):
            raise RuntimeError(f"Invalid QED result: {qed_result}")
        return float(qed_result)
    except Exception as e:
        raise RuntimeError(f"QED calculation failed: {e}")


def compute_sas(mol: Chem.Mol) -> float:
    """Compute SAS (Synthetic Accessibility Score). Raises exception on failure."""
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    try:
        sas_result = calculate_sa_score(mol)  # Use existing function from mol_scores.py
        if sas_result is None or not isinstance(sas_result, (int, float)):
            raise RuntimeError(f"Invalid SAS result: {sas_result}")
        return float(sas_result)
    except Exception as e:
        raise RuntimeError(f"SAS calculation failed: {e}")


def sigmoid_scaling(x: float, slope: float, inflection: float) -> float:
    """Scales a value using a sigmoid function, where lower x gives higher output."""
    return 1 / (1 + np.exp(slope * (x - inflection)))


def compute_three_criterion_reward(mol: Chem.Mol, 
                                 docking_score: Optional[float] = None,
                                 pdb_id: Optional[str] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 use_binding_affinity: bool = True,
                                 affinity_config: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward based on three criteria: QED, SAS, and Binding Affinity.
    Binding affinity gets highest weight as primary optimization criterion.
    
    ALL THREE CRITERIA MUST SUCCEED OR THE FUNCTION WILL RAISE AN EXCEPTION.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional pre-computed docking score
        pdb_id: PDB ID for GNINA docking
        weights: Optional weights for the three criteria
        use_binding_affinity: Whether to use binding affinity
        affinity_config: Configuration for sigmoid scaling of binding affinity.
                         Example: {'slope': 1.0, 'inflection': -9.0}
        
    Returns:
        Tuple of (reward, metrics_dict)
        
    Raises:
        ValueError: If molecule is invalid
        RuntimeError: If any criterion calculation fails
    """
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    # Default weights - binding affinity is primary criterion
    default_weights = {
        'qed': 1.0,
        'sas': 1.0,
        'binding_affinity': 3.0  # Higher weight for binding affinity
    }
    
    if weights is not None:
        default_weights.update(weights)
    w = default_weights
    
    # Configuration for binding affinity normalization
    affinity_conf = affinity_config or {'slope': 1.0, 'inflection': -9.0}
    
    # Get SMILES for docking
    smiles = Chem.MolToSmiles(mol)
    
    # Initialize components
    components = {}
    total_weight = 0
    reward = 0.0
    
    # 1. Compute QED (0-1, higher is better) - REQUIRED
    qed_score_val = compute_qed(mol)
    qed_component = qed_score_val * w['qed']
    components['qed'] = qed_component
    reward += qed_component
    total_weight += w['qed']
    
    # 2. Compute SAS (1-10, lower is better, so invert to 0-1 scale) - REQUIRED
    sas_score_val = compute_sas(mol)
    sas_normalized = max(0, (10 - sas_score_val) / 9)  # Invert: 1->1, 10->0
    sas_component = sas_normalized * w['sas']
    components['sas'] = sas_component
    reward += sas_component
    total_weight += w['sas']
    
    # 3. Compute binding affinity (more negative is better) - REQUIRED if use_binding_affinity
    binding_affinity = None
    if use_binding_affinity:
        if docking_score is not None:
            # Use pre-computed docking score
            binding_affinity = docking_score
        elif pdb_id is not None:
            # Compute binding affinity using GNINA
            binding_affinity = compute_binding_affinity(pdb_id, smiles)
        else:
            raise ValueError("Binding affinity requested but no docking_score or pdb_id provided")
        
        # Convert binding affinity to 0-1 scale using a sigmoid function
        # This provides a non-linear reward, strongly incentivizing values below the inflection point
        binding_normalized = sigmoid_scaling(
            binding_affinity, 
            slope=affinity_conf['slope'], 
            inflection=affinity_conf['inflection']
        )
        binding_component = binding_normalized * w['binding_affinity']
        components['binding_affinity'] = binding_component
        reward += binding_component
        total_weight += w['binding_affinity']
    
    # Normalize reward by total weight
    if total_weight > 0:
        reward = reward / total_weight
    else:
        raise RuntimeError("No valid criteria computed")
    
    # Determine which criteria were successfully computed
    criteria_used = ['qed', 'sas']
    if binding_affinity is not None:
        criteria_used.append('binding_affinity')
    
    # Compile metrics
    metrics = {
        'smiles': smiles,
        'qed': qed_score_val,
        'sas': sas_score_val,
        'sas_normalized': sas_normalized,
        'binding_affinity': binding_affinity,
        'binding_affinity_normalized': binding_normalized if binding_affinity is not None else None,
        'qed_component': components.get('qed', 0.0),
        'sas_component': components.get('sas', 0.0),
        'binding_component': components.get('binding_affinity', 0.0),
        'reward': reward,
        'criteria_used': criteria_used,
        'total_weight': total_weight
    }
    
    return reward, metrics


def compute_reward(mol: Chem.Mol, 
                  docking_score: Optional[float] = None,
                  pdb_id: Optional[str] = None,
                  use_binding_affinity: bool = True,
                  **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Simplified reward function with only 3 criteria: QED, SAS, Binding Affinity.
    Binding affinity is the primary optimization criterion.
    
    ALL THREE CRITERIA MUST SUCCEED OR THE FUNCTION WILL RAISE AN EXCEPTION.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional pre-computed docking score
        pdb_id: PDB ID for GNINA docking
        use_binding_affinity: Whether to use binding affinity
        **kwargs: Additional arguments (for backward compatibility)
        
    Returns:
        Tuple of (reward, metrics_dict)
        
    Raises:
        ValueError: If molecule is invalid
        RuntimeError: If any criterion calculation fails
    """
    # Extract weights and affinity_config from kwargs if provided
    weights = kwargs.get('weights', None)
    affinity_config = kwargs.get('affinity_config', None)
    
    return compute_three_criterion_reward(
        mol=mol,
        docking_score=docking_score,
        pdb_id=pdb_id,
        weights=weights,
        use_binding_affinity=use_binding_affinity,
        affinity_config=affinity_config
    )


def batch_compute_rewards(mols: List[Chem.Mol], 
                         pdb_id: Optional[str] = None,
                         docking_scores: Optional[List[float]] = None,
                         use_binding_affinity: bool = True,
                         **kwargs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Compute rewards for a batch of molecules using 3 criteria.
    
    ALL THREE CRITERIA MUST SUCCEED FOR EACH MOLECULE OR THE FUNCTION WILL RAISE AN EXCEPTION.
    
    Args:
        mols: List of RDKit molecule objects
        pdb_id: PDB ID for GNINA docking
        docking_scores: Optional list of pre-computed docking scores
        use_binding_affinity: Whether to compute binding affinity
        **kwargs: Additional arguments for reward computation
        
    Returns:
        Tuple of (rewards_list, metrics_list)
        
    Raises:
        ValueError: If any molecule is invalid
        RuntimeError: If any criterion calculation fails
    """
    rewards = []
    metrics_list = []
    
    # Show progress for large batches
    if len(mols) > 10:
        logging.info(f"Computing rewards for {len(mols)} molecules with binding affinity: {use_binding_affinity}")
    
    for i, mol in enumerate(mols):
        dock_score = docking_scores[i] if docking_scores is not None else None
        
        try:
            reward, metrics = compute_reward(
                mol=mol, 
                docking_score=dock_score,
                pdb_id=pdb_id,
                use_binding_affinity=use_binding_affinity,
                **kwargs
            )
            
            rewards.append(reward)
            metrics_list.append(metrics)
            
            # Log progress for docking
            if use_binding_affinity and metrics.get('binding_affinity') is not None:
                if i % 10 == 0 and len(mols) > 10:
                    logging.info(f"Processed {i+1}/{len(mols)} molecules with docking")
        
        except Exception as e:
            raise RuntimeError(f"Failed to compute reward for molecule {i}: {e}")
    
    return rewards, metrics_list


def analyze_reward_distribution(rewards: List[float], 
                               metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of rewards and their three components.
    
    Args:
        rewards: List of reward values
        metrics_list: List of metrics dictionaries
        
    Returns:
        Analysis dictionary
        
    Raises:
        ValueError: If input data is invalid
    """
    if not rewards or not metrics_list:
        raise ValueError("Empty rewards or metrics list provided")
    
    if len(rewards) != len(metrics_list):
        raise ValueError(f"Length mismatch: rewards={len(rewards)}, metrics={len(metrics_list)}")
    
    rewards_array = np.array(rewards)
    
    # Basic statistics
    analysis = {
        'reward_stats': {
            'mean': float(np.mean(rewards_array)),
            'std': float(np.std(rewards_array)),
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'median': float(np.median(rewards_array))
        }
    }
    
    # Component statistics for the three criteria
    components = ['qed', 'sas', 'binding_affinity']
    for comp in components:
        comp_values = [m[comp] for m in metrics_list if comp in m and m[comp] is not None]
        if comp_values:
            analysis[f'{comp}_stats'] = {
                'mean': float(np.mean(comp_values)),
                'std': float(np.std(comp_values)),
                'min': float(np.min(comp_values)),
                'max': float(np.max(comp_values))
            }
    
    # Binding affinity specific statistics
    binding_affinities = [m.get('binding_affinity') for m in metrics_list]
    binding_affinities = [x for x in binding_affinities if x is not None]
    
    if binding_affinities:
        analysis['binding_affinity_success_rate'] = len(binding_affinities) / len(metrics_list)
        if 'binding_affinity_stats' not in analysis:
            analysis['binding_affinity_stats'] = {
                'mean': float(np.mean(binding_affinities)),
                'std': float(np.std(binding_affinities)),
                'min': float(np.min(binding_affinities)),
                'max': float(np.max(binding_affinities))
            }
        analysis['binding_affinity_stats']['best_binding'] = float(np.min(binding_affinities))  # Most negative = best
    else:
        analysis['binding_affinity_success_rate'] = 0.0
    
    # Criteria availability statistics
    criteria_counts = {'qed': 0, 'sas': 0, 'binding_affinity': 0}
    for metrics in metrics_list:
        criteria_used = metrics.get('criteria_used', [])
        for criterion in criteria_used:
            if criterion in criteria_counts:
                criteria_counts[criterion] += 1
    
    analysis['criteria_availability'] = {
        criterion: count / len(metrics_list) 
        for criterion, count in criteria_counts.items()
    }
    
    # Correlation analysis
    if len(rewards) > 5:
        qed_values = [m.get('qed', 0) for m in metrics_list if m.get('qed') is not None]
        sas_values = [m.get('sas', 0) for m in metrics_list if m.get('sas') is not None]
        
        analysis['correlations'] = {}
        
        if len(qed_values) > 1 and len(set(qed_values)) > 1:
            qed_rewards = [rewards[i] for i, m in enumerate(metrics_list) if m.get('qed') is not None]
            if len(qed_rewards) == len(qed_values):
                analysis['correlations']['reward_qed'] = float(np.corrcoef(qed_rewards, qed_values)[0, 1])
        
        if len(sas_values) > 1 and len(set(sas_values)) > 1:
            sas_rewards = [rewards[i] for i, m in enumerate(metrics_list) if m.get('sas') is not None]
            if len(sas_rewards) == len(sas_values):
                analysis['correlations']['reward_sas'] = float(np.corrcoef(sas_rewards, sas_values)[0, 1])
        
        if binding_affinities and len(binding_affinities) > 5:
            # Only compute correlation for molecules with binding affinity
            binding_indices = [i for i, m in enumerate(metrics_list) if m.get('binding_affinity') is not None]
            binding_rewards = [rewards[i] for i in binding_indices]
            
            if len(binding_rewards) > 1:
                analysis['correlations']['reward_binding'] = float(
                    np.corrcoef(binding_rewards, binding_affinities)[0, 1]
                )
    
    return analysis