import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import GNINA docking function
from fairseq.molecule_utils.basic.run_gnina_docking import docking

# Import existing scoring functions from mol_scores.py
from fairseq.molecule_utils.basic.mol_scores import qed_score, calculate_sa_score
from rdkit import Chem


def compute_binding_affinity(pdb_id: str, smiles: str, max_retries: int = 3) -> float:
    """
    Computes binding affinity for a given SMILES string and PDB ID using GNINA docking.
    
    This function will attempt to dock the molecule multiple times and raises an
    exception if all attempts fail.
    """
    if not smiles or not pdb_id:
        raise ValueError(f"Invalid input: pdb_id='{pdb_id}', smiles='{smiles}'")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    for attempt in range(max_retries):
        try:
            affinity = docking(pdb_id=pdb_id, ligand_smiles=smiles)
            
            if affinity is not None and isinstance(affinity, (int, float)):
                if -20.0 <= affinity <= 5.0:
                    return float(affinity)
            
            raise RuntimeError(f"Invalid binding affinity result: {affinity}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Docking attempt {attempt + 1} failed for {smiles[:20]}...: {e}")
                continue
            else:
                raise RuntimeError(f"All {max_retries} docking attempts failed for {smiles}: {e}")
    
    raise RuntimeError(f"Docking failed to produce a result after {max_retries} attempts.")


def compute_qed(mol: Chem.Mol) -> float:
    """Calculates the Quantitative Estimate of Drug-likeness (QED) for a molecule."""
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    try:
        qed_result = qed_score(mol)
        if qed_result is None or not isinstance(qed_result, (int, float)):
            raise RuntimeError(f"Invalid QED result: {qed_result}")
        return float(qed_result)
    except Exception as e:
        raise RuntimeError(f"QED calculation failed: {e}")


def compute_sas(mol: Chem.Mol) -> float:
    """Calculates the Synthetic Accessibility Score (SAS) for a molecule."""
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    try:
        sas_result = calculate_sa_score(mol)
        if sas_result is None or not isinstance(sas_result, (int, float)):
            raise RuntimeError(f"Invalid SAS result: {sas_result}")
        return float(sas_result)
    except Exception as e:
        raise RuntimeError(f"SAS calculation failed: {e}")


def sigmoid_scaling(x: float, slope: float, inflection: float) -> float:
    """Applies a sigmoid scaling function, mapping lower input values to higher output values."""
    return 1 / (1 + np.exp(slope * (x - inflection)))


def compute_three_criterion_reward(mol: Chem.Mol, 
                                 docking_score: Optional[float] = None,
                                 pdb_id: Optional[str] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 use_binding_affinity: bool = True,
                                 affinity_config: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculates a composite reward based on QED, SAS, and Binding Affinity.
    
    This function requires all three metrics to be successfully computed and gives
    binding affinity the highest weight by default. It will raise an exception if any
    of the underlying calculations fail.
    
    Args:
        mol: The RDKit molecule object.
        docking_score: An optional pre-computed docking score.
        pdb_id: The PDB ID of the target for GNINA docking.
        weights: Optional weights for the three criteria.
        use_binding_affinity: If True, enables binding affinity as an optimization criterion.
        affinity_config: Configuration for the sigmoid scaling of binding affinity,
                         e.g., {'slope': 1.0, 'inflection': -9.0}.
        
    Returns:
        A tuple containing the final reward and a dictionary of detailed metrics.
    """
    if mol is None:
        raise ValueError("Invalid molecule: None")
    
    default_weights = {
        'qed': 1.0,
        'sas': 1.0,
        'binding_affinity': 3.0
    }
    
    if weights is not None:
        default_weights.update(weights)
    w = default_weights
    
    affinity_conf = affinity_config or {'slope': 1.0, 'inflection': -9.0}
    
    smiles = Chem.MolToSmiles(mol)
    
    components = {}
    total_weight = 0
    reward = 0.0
    
    qed_score_val = compute_qed(mol)
    qed_component = qed_score_val * w['qed']
    components['qed'] = qed_component
    reward += qed_component
    total_weight += w['qed']
    
    sas_score_val = compute_sas(mol)
    sas_normalized = max(0, (10 - sas_score_val) / 9)
    sas_component = sas_normalized * w['sas']
    components['sas'] = sas_component
    reward += sas_component
    total_weight += w['sas']
    
    binding_affinity = None
    binding_normalized = None
    if use_binding_affinity:
        if docking_score is not None:
            binding_affinity = docking_score
        elif pdb_id is not None:
            binding_affinity = compute_binding_affinity(pdb_id, smiles)
        else:
            raise ValueError("Binding affinity requested but no docking_score or pdb_id provided")
        
        binding_normalized = sigmoid_scaling(
            binding_affinity, 
            slope=affinity_conf['slope'], 
            inflection=affinity_conf['inflection']
        )
        binding_component = binding_normalized * w['binding_affinity']
        components['binding_affinity'] = binding_component
        reward += binding_component
        total_weight += w['binding_affinity']
    
    if total_weight > 0:
        reward = reward / total_weight
    else:
        raise RuntimeError("No valid criteria were computed, so total weight is zero.")
    
    criteria_used = ['qed', 'sas']
    if binding_affinity is not None:
        criteria_used.append('binding_affinity')
    
    metrics = {
        'smiles': smiles,
        'qed': qed_score_val,
        'sas': sas_score_val,
        'sas_normalized': sas_normalized,
        'binding_affinity': binding_affinity,
        'binding_affinity_normalized': binding_normalized,
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
    A wrapper for `compute_three_criterion_reward` to calculate a composite score.
    
    This function requires QED, SAS, and optionally binding affinity to be successfully
    computed, otherwise it will raise an exception.
    
    Args:
        mol: The RDKit molecule object.
        docking_score: An optional pre-computed docking score.
        pdb_id: The PDB ID for GNINA docking.
        use_binding_affinity: If True, enables binding affinity as an optimization criterion.
        **kwargs: Additional arguments for backward compatibility (e.g., weights).
        
    Returns:
        A tuple containing the final reward and a dictionary of detailed metrics.
    """
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
    Computes rewards for a batch of molecules.
    
    This function requires successful calculation of all specified criteria for
    every molecule in the batch, otherwise it will raise an exception.
    
    Args:
        mols: A list of RDKit molecule objects.
        pdb_id: The PDB ID for GNINA docking.
        docking_scores: An optional list of pre-computed docking scores.
        use_binding_affinity: If True, enables binding affinity as an optimization criterion.
        **kwargs: Additional arguments for the reward computation.
        
    Returns:
        A tuple containing a list of rewards and a list of metric dictionaries.
    """
    rewards = []
    metrics_list = []
    
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
            
            if use_binding_affinity and metrics.get('binding_affinity') is not None:
                if i % 10 == 0 and len(mols) > 10:
                    logging.info(f"Processed {i+1}/{len(mols)} molecules with docking")
        
        except Exception as e:
            raise RuntimeError(f"Failed to compute reward for molecule {i}: {e}")
    
    return rewards, metrics_list


def analyze_reward_distribution(rewards: List[float], 
                               metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes the statistical distribution of rewards and their underlying components
    (e.g., QED, SAS, and binding affinity).
    
    Args:
        rewards: A list of reward values.
        metrics_list: A list of metrics dictionaries from the reward computation.
        
    Returns:
        A dictionary containing the statistical analysis.
        
    Raises:
        ValueError: If the input lists are empty or have mismatched lengths.
    """
    if not rewards or not metrics_list:
        raise ValueError("Empty rewards or metrics list provided")
    
    if len(rewards) != len(metrics_list):
        raise ValueError(f"Length mismatch: rewards={len(rewards)}, metrics={len(metrics_list)}")
    
    rewards_array = np.array(rewards)
    
    analysis: Dict[str, Any] = {
        'reward_stats': {
            'mean': float(np.mean(rewards_array)),
            'std': float(np.std(rewards_array)),
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'median': float(np.median(rewards_array))
        }
    }
    
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
        analysis['binding_affinity_stats']['best_binding'] = float(np.min(binding_affinities))
    else:
        analysis['binding_affinity_success_rate'] = 0.0
    
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
            binding_indices = [i for i, m in enumerate(metrics_list) if m.get('binding_affinity') is not None]
            binding_rewards = [rewards[i] for i in binding_indices]
            
            if len(binding_rewards) > 1:
                analysis['correlations']['reward_binding'] = float(
                    np.corrcoef(binding_rewards, binding_affinities)[0, 1]
                )
    
    return analysis