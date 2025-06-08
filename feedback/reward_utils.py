"""
Improved reward computation for molecular optimization.
Includes better scaling, diversity bonuses, and multi-objective optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs
import math

# Try to import rdMolDescriptors, use fallback if not available
try:
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    rdMolDescriptors = None


def compute_qed(mol: Chem.Mol) -> float:
    """
    Compute QED (Quantitative Estimation of Drug-likeness) for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        QED score (0-1, higher is better)
    """
    try:
        return QED.qed(mol)
    except Exception as e:
        print(f"[WARNING] QED calculation failed: {e}")
        return 0.0


def compute_sas(mol: Chem.Mol) -> float:
    """
    Compute SAS (Synthetic Accessibility Score) for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        SAS score (1-10, lower is better/more synthesizable)
    """
    if mol is None:
        return 10.0
    
    if rdMolDescriptors is not None:
        # Try different function names that exist in different RDKit versions
        sas_functions = [
            'CalcSyntheticAccessibilityScore',
            'BertzCT'  # Fallback complexity measure
        ]
        
        for func_name in sas_functions:
            if hasattr(rdMolDescriptors, func_name):
                try:
                    func = getattr(rdMolDescriptors, func_name)
                    score = func(mol)
                    
                    if func_name == 'CalcSyntheticAccessibilityScore':
                        return float(score)
                    elif func_name == 'BertzCT':
                        # Normalize BertzCT to 1-10 range
                        return float(min(10, max(1, 1 + score / 100)))
                        
                except Exception:
                    continue
    
    # Simple fallback based on molecular complexity
    try:
        complexity = (mol.GetNumHeavyAtoms() * 0.1 + 
                     Descriptors.RingCount(mol) * 0.5 + 
                     Descriptors.NumHeteroatoms(mol) * 0.2)
        return float(min(10, max(1, 1 + complexity)))
    except Exception:
        return 6.0  # Default middle value


def compute_lipinski_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute Lipinski Rule of Five descriptors.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with Lipinski descriptors
    """
    try:
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
    except Exception as e:
        print(f"[WARNING] Lipinski descriptors calculation failed: {e}")
        return {
            'mw': 1000.0, 'logp': 10.0, 'hbd': 20, 
            'hba': 20, 'rotatable_bonds': 20, 'tpsa': 200.0
        }


def compute_diversity_bonus(mol: Chem.Mol, reference_mols: List[Chem.Mol], 
                          similarity_threshold: float = 0.7) -> float:
    """
    Compute diversity bonus based on Tanimoto similarity to reference molecules.
    
    Args:
        mol: Query molecule
        reference_mols: List of reference molecules
        similarity_threshold: Threshold below which diversity bonus is given
        
    Returns:
        Diversity bonus (0-1, higher is more diverse)
    """
    if not reference_mols or mol is None:
        return 0.0
    
    try:
        query_fp = FingerprintMols.FingerprintMol(mol)
        similarities = []
        
        for ref_mol in reference_mols:
            if ref_mol is not None:
                ref_fp = FingerprintMols.FingerprintMol(ref_mol)
                sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        max_similarity = max(similarities)
        
        # Give bonus for molecules that are dissimilar to existing ones
        if max_similarity < similarity_threshold:
            diversity_bonus = (similarity_threshold - max_similarity) / similarity_threshold
            return min(diversity_bonus, 0.5)  # Cap at 0.5
        
        return 0.0
        
    except Exception as e:
        print(f"[WARNING] Diversity bonus calculation failed: {e}")
        return 0.0


def compute_advanced_reward(mol: Chem.Mol, 
                          docking_score: Optional[float] = None,
                          target_properties: Optional[Dict[str, float]] = None,
                          reference_mols: Optional[List[Chem.Mol]] = None,
                          weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Advanced reward computation with multiple objectives and proper scaling.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional docking score (lower is better)
        target_properties: Optional target property values
        reference_mols: Optional reference molecules for diversity
        weights: Optional weights for different components
        
    Returns:
        Tuple of (reward, metrics_dict)
    """
    if mol is None:
        return -10.0, {"error": "invalid_molecule"}
    
    # Default weights
    default_weights = {
        'qed': 2.0,
        'sas': 1.0,
        'lipinski': 1.5,
        'diversity': 0.5,
        'docking': 2.0,
        'target_similarity': 1.0
    }
    additional_penalties = 0
    
    if weights is not None:
        default_weights.update(weights)
    w = default_weights
    
    # Compute basic descriptors
    qed_score = compute_qed(mol)
    sas_score = compute_sas(mol)
    lipinski_desc = compute_lipinski_descriptors(mol)
    
    # QED component (0-1, higher is better)
    qed_component = qed_score * w['qed']
    
    # SAS component (1-10 -> 0-1, higher is better)
    sas_normalized = max(0, (10 - sas_score) / 9)  # Invert and normalize
    sas_component = sas_normalized * w['sas']
    
    # Lipinski Rule of Five component
    lipinski_violations = sum([
        lipinski_desc['mw'] > 500,
        lipinski_desc['logp'] > 5,
        lipinski_desc['hbd'] > 5,
        lipinski_desc['hba'] > 10
    ])
    
    # Penalize violations but don't make it too harsh
    lipinski_penalty = min(lipinski_violations * 0.3, 1.0)
    lipinski_component = (1 - lipinski_penalty) * w['lipinski']
    
    # Additional molecular properties
    # LogP should be in reasonable range (0-5)
    sas_normalized = max(0, min(1, (10 - sas_score) / 9))
    sas_component = sas_normalized * w['sas']

    # LogP should be in reasonable range (0-5) - add reward for being in range
    logp_reward = 0
    if 0 <= lipinski_desc['logp'] <= 5:
        # Bonus for being within ideal range
        logp_reward = 1.0 - abs(lipinski_desc['logp'] - 2.5) / 2.5  # Centered around 2.5
    else: 
        logp_reward = -1.0
    logp_component = logp_reward * w['logp']

    # Docking component (handle minimization)
    docking_component = 0
    if docking_score is not None:
        # Convert minimization to maximization: -docking_score
        # Clip and normalize negative docking scores (assumes scores are negative)
        docking_normalized = max(0, min(1, (-docking_score) / 15))  # Normalize to 0-1
        docking_component = docking_normalized * w['docking']

    # Diversity component (use weight directly)
    diversity_bonus = 0
    if reference_mols is not None:
        diversity_bonus = compute_diversity_bonus(mol, reference_mols) * w['diversity']
    
    # Docking component
    docking_component = 0
    if docking_score is not None:
        # Assuming docking score is negative (lower is better)
        # Convert to positive reward (higher is better)
        docking_normalized = max(0, min(-docking_score / 10, 1))  # Normalize to 0-1
        docking_component = docking_normalized * w['docking']
    
    # Target property similarity component
    target_component = 0
    if target_properties is not None:
        target_similarities = []
        
        if 'mw' in target_properties:
            mw_diff = abs(lipinski_desc['mw'] - target_properties['mw'])
            mw_sim = max(0, 1 - mw_diff / 200)  # Similarity within 200 Da
            target_similarities.append(mw_sim)
        
        if 'logp' in target_properties:
            logp_diff = abs(lipinski_desc['logp'] - target_properties['logp'])
            logp_sim = max(0, 1 - logp_diff / 3)  # Similarity within 3 logP units
            target_similarities.append(logp_sim)
        
        if target_similarities:
            target_component = np.mean(target_similarities) * w['target_similarity']
    
    # Combine all components
    reward = (qed_component + 
              sas_component + 
              lipinski_component + 
              diversity_bonus + 
              docking_component + 
              target_component - 
              additional_penalties)
    
    # Compile detailed metrics
    metrics = {
        'qed': qed_score,
        'sas': sas_score,
        'sas_normalized': sas_normalized,
        'mw': lipinski_desc['mw'],
        'logp': lipinski_desc['logp'],
        'hbd': lipinski_desc['hbd'],
        'hba': lipinski_desc['hba'],
        'rotatable_bonds': lipinski_desc['rotatable_bonds'],
        'tpsa': lipinski_desc['tpsa'],
        'lipinski_violations': lipinski_violations,
        'qed_component': qed_component,
        'sas_component': sas_component,
        'logp_component': logp_component,
        'diversity_bonus': diversity_bonus,
        'docking_component': docking_component,
        'target_component': target_component,
        'additional_penalties': additional_penalties,
        'reward': reward
    }
    
    return reward, metrics


def compute_reward(mol: Chem.Mol, 
                  docking_score: Optional[float] = None,
                  lambda_sas: float = 0.3,
                  lambda_logp: float = 0.1,
                  lambda_mw: float = 0.1,
                  **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Backward-compatible reward function with improved implementation.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional docking score
        lambda_sas: Weight for SAS penalty
        lambda_logp: Weight for LogP penalty  
        lambda_mw: Weight for MW penalty
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (reward, metrics_dict)
    """
    # Use the advanced reward function with converted weights
    weights = {
        'qed': 2.0,
        'sas': lambda_sas * 3.0,  # Convert to positive weight
        'lipinski': 1.0,
        'docking': 1.0 if docking_score is not None else 0.0
    }
    
    return compute_advanced_reward(
        mol=mol,
        docking_score=docking_score,
        weights=weights,
        **kwargs
    )


def batch_compute_rewards(mols: List[Chem.Mol], 
                         docking_scores: Optional[List[float]] = None,
                         **kwargs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Compute rewards for a batch of molecules efficiently.
    
    Args:
        mols: List of RDKit molecule objects
        docking_scores: Optional list of docking scores
        **kwargs: Additional arguments for reward computation
        
    Returns:
        Tuple of (rewards_list, metrics_list)
    """
    rewards = []
    metrics_list = []
    
    for i, mol in enumerate(mols):
        dock_score = docking_scores[i] if docking_scores is not None else None
        reward, metrics = compute_reward(mol, docking_score=dock_score, **kwargs)
        rewards.append(reward)
        metrics_list.append(metrics)
    
    return rewards, metrics_list


def analyze_reward_distribution(rewards: List[float], 
                               metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of rewards and their components.
    
    Args:
        rewards: List of reward values
        metrics_list: List of metrics dictionaries
        
    Returns:
        Analysis dictionary
    """
    if not rewards or not metrics_list:
        return {}
    
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
    
    # Component statistics
    components = ['qed', 'sas', 'mw', 'logp', 'lipinski_violations']
    for comp in components:
        if comp in metrics_list[0]:
            comp_values = [m[comp] for m in metrics_list if comp in m]
            if comp_values:
                analysis[f'{comp}_stats'] = {
                    'mean': float(np.mean(comp_values)),
                    'std': float(np.std(comp_values)),
                    'min': float(np.min(comp_values)),
                    'max': float(np.max(comp_values))
                }
    
    # Correlation analysis
    if len(rewards) > 5:
        qed_values = [m.get('qed', 0) for m in metrics_list]
        sas_values = [m.get('sas', 0) for m in metrics_list]
        
        analysis['correlations'] = {
            'reward_qed': float(np.corrcoef(rewards, qed_values)[0, 1]) if len(set(qed_values)) > 1 else 0,
            'reward_sas': float(np.corrcoef(rewards, sas_values)[0, 1]) if len(set(sas_values)) > 1 else 0
        }
    
    return analysis