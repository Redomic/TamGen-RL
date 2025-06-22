"""
Improved reward computation for molecular optimization with GNINA docking.
Includes binding affinity as primary criterion with proper scaling and multi-objective optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs
import math
import logging

# Import GNINA docking function
try:
    from fairseq.molecule_utils.basic.run_gnina_docking import docking
    GNINA_AVAILABLE = True
except ImportError:
    GNINA_AVAILABLE = False
    logging.warning("GNINA docking not available. Binding affinity will be disabled.")

# Try to import rdMolDescriptors, use fallback if not available
try:
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    rdMolDescriptors = None


def compute_binding_affinity(pdb_id: str, smiles: str, max_retries: int = 3) -> Optional[float]:
    """
    Compute binding affinity using GNINA docking.
    
    Args:
        pdb_id: PDB ID of the target protein
        smiles: SMILES string of the ligand
        max_retries: Maximum number of retry attempts
        
    Returns:
        Binding affinity in kcal/mol (more negative = better binding) or None if failed
    """
    if not GNINA_AVAILABLE:
        logging.warning("GNINA not available, returning None for binding affinity")
        return None
    
    if not smiles or not pdb_id:
        return None
    
    # Validate SMILES first
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    for attempt in range(max_retries):
        try:
            affinity = docking(pdb_id=pdb_id, ligand_smiles=smiles)
            
            # Validate the result
            if affinity is not None and isinstance(affinity, (int, float)):
                # Reasonable range check for binding affinity (-20 to +5 kcal/mol)
                if -20.0 <= affinity <= 5.0:
                    return float(affinity)
                else:
                    logging.warning(f"Binding affinity out of range: {affinity} kcal/mol")
                    return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Docking attempt {attempt + 1} failed for {smiles[:20]}...: {e}")
                continue
            else:
                logging.error(f"All docking attempts failed for {smiles[:20]}...: {e}")
                return None
    
    return None


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
        logging.warning(f"QED calculation failed: {e}")
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
        logging.warning(f"Lipinski descriptors calculation failed: {e}")
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
        logging.warning(f"Diversity bonus calculation failed: {e}")
        return 0.0


def compute_advanced_reward(mol: Chem.Mol, 
                          docking_score: Optional[float] = None,
                          pdb_id: Optional[str] = None,
                          target_properties: Optional[Dict[str, float]] = None,
                          reference_mols: Optional[List[Chem.Mol]] = None,
                          weights: Optional[Dict[str, float]] = None,
                          use_binding_affinity: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Advanced reward computation with binding affinity as primary criterion.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional pre-computed docking score (lower is better)
        pdb_id: PDB ID for GNINA docking (if docking_score not provided)
        target_properties: Optional target property values
        reference_mols: Optional reference molecules for diversity
        weights: Optional weights for different components
        use_binding_affinity: Whether to compute binding affinity
        
    Returns:
        Tuple of (reward, metrics_dict)
    """
    if mol is None:
        return -10.0, {"error": "invalid_molecule"}
    
    # Default weights - binding affinity dominates when available
    default_weights = {
        'binding_affinity': 10.0,  # Primary criterion - no scaling needed
        'qed': 1.0,
        'sas': 0.5,
        'lipinski': 0.8,
        'diversity': 0.3,
        'target_similarity': 0.5
    }
    
    if weights is not None:
        default_weights.update(weights)
    w = default_weights
    
    # Get SMILES for docking
    smiles = Chem.MolToSmiles(mol)
    
    # Compute binding affinity (primary criterion)
    binding_affinity = None
    binding_component = 0.0
    
    if use_binding_affinity and GNINA_AVAILABLE:
        if docking_score is not None:
            # Use pre-computed docking score
            binding_affinity = docking_score
        elif pdb_id is not None:
            # Compute binding affinity using GNINA
            binding_affinity = compute_binding_affinity(pdb_id, smiles)
        
        if binding_affinity is not None:
            # Convert binding affinity to reward component
            # More negative = better binding = higher reward
            # Scale: -12 kcal/mol -> 1.0, -6 kcal/mol -> 0.5, 0 kcal/mol -> 0.0
            binding_component = max(0.0, min(1.0, (-binding_affinity) / 12.0)) * w['binding_affinity']
    
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
    lipinski_penalty = min(lipinski_violations * 0.2, 0.8)
    lipinski_component = (1 - lipinski_penalty) * w['lipinski']
    
    # Additional molecular properties
    # LogP should be in reasonable range (0-5)
    logp_penalty = 0
    if lipinski_desc['logp'] < 0:
        logp_penalty = abs(lipinski_desc['logp']) * 0.05
    elif lipinski_desc['logp'] > 5:
        logp_penalty = (lipinski_desc['logp'] - 5) * 0.05
    
    # Molecular weight penalty for very large molecules
    mw_penalty = max(0, (lipinski_desc['mw'] - 600) / 400) * 0.1
    
    # TPSA should be reasonable (20-140)
    tpsa_penalty = 0
    if lipinski_desc['tpsa'] > 140:
        tpsa_penalty = (lipinski_desc['tpsa'] - 140) / 100 * 0.05
    elif lipinski_desc['tpsa'] < 20:
        tpsa_penalty = (20 - lipinski_desc['tpsa']) / 20 * 0.05
    
    # Combine penalties
    additional_penalties = logp_penalty + mw_penalty + tpsa_penalty
    
    # Diversity component
    diversity_bonus = 0
    if reference_mols is not None:
        diversity_bonus = compute_diversity_bonus(mol, reference_mols) * w['diversity']
    
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
    # When binding affinity is available, it dominates the reward
    if binding_affinity is not None:
        # Binding affinity is the primary criterion
        reward = (binding_component + 
                 qed_component * 0.3 +  # Reduced weight for other components
                 sas_component * 0.3 + 
                 lipinski_component * 0.3 + 
                 diversity_bonus + 
                 target_component - 
                 additional_penalties)
    else:
        # Fallback to standard multi-objective optimization
        reward = (qed_component + 
                 sas_component + 
                 lipinski_component + 
                 diversity_bonus + 
                 target_component - 
                 additional_penalties)
    
    # Compile detailed metrics
    metrics = {
        'smiles': smiles,
        'binding_affinity': binding_affinity,
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
        'binding_component': binding_component,
        'qed_component': qed_component,
        'sas_component': sas_component,
        'lipinski_component': lipinski_component,
        'diversity_bonus': diversity_bonus,
        'target_component': target_component,
        'additional_penalties': additional_penalties,
        'reward': reward,
        'has_binding_affinity': binding_affinity is not None
    }
    
    return reward, metrics


def compute_reward(mol: Chem.Mol, 
                  docking_score: Optional[float] = None,
                  pdb_id: Optional[str] = None,
                  lambda_sas: float = 0.3,
                  lambda_logp: float = 0.1,
                  lambda_mw: float = 0.1,
                  use_binding_affinity: bool = True,
                  **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Backward-compatible reward function with binding affinity support.
    
    Args:
        mol: RDKit molecule object
        docking_score: Optional pre-computed docking score
        pdb_id: PDB ID for GNINA docking
        lambda_sas: Weight for SAS penalty
        lambda_logp: Weight for LogP penalty  
        lambda_mw: Weight for MW penalty
        use_binding_affinity: Whether to use binding affinity
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (reward, metrics_dict)
    """
    # Use the advanced reward function with converted weights
    weights = {
        'binding_affinity': 10.0 if use_binding_affinity else 0.0,
        'qed': 2.0,
        'sas': lambda_sas * 3.0,  # Convert to positive weight
        'lipinski': 1.0,
    }
    
    return compute_advanced_reward(
        mol=mol,
        docking_score=docking_score,
        pdb_id=pdb_id,
        weights=weights,
        use_binding_affinity=use_binding_affinity,
        **kwargs
    )


def batch_compute_rewards_with_docking(mols: List[Chem.Mol], 
                                     pdb_id: Optional[str] = None,
                                     docking_scores: Optional[List[float]] = None,
                                     use_binding_affinity: bool = True,
                                     **kwargs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Compute rewards for a batch of molecules with optional docking.
    
    Args:
        mols: List of RDKit molecule objects
        pdb_id: PDB ID for GNINA docking
        docking_scores: Optional list of pre-computed docking scores
        use_binding_affinity: Whether to compute binding affinity
        **kwargs: Additional arguments for reward computation
        
    Returns:
        Tuple of (rewards_list, metrics_list)
    """
    rewards = []
    metrics_list = []
    
    # Show progress for large batches
    if len(mols) > 10:
        logging.info(f"Computing rewards for {len(mols)} molecules with binding affinity: {use_binding_affinity}")
    
    for i, mol in enumerate(mols):
        dock_score = docking_scores[i] if docking_scores is not None else None
        
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
    
    return rewards, metrics_list


def batch_compute_rewards(mols: List[Chem.Mol], 
                         docking_scores: Optional[List[float]] = None,
                         **kwargs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Legacy function - backward compatibility.
    
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
    components = ['qed', 'sas', 'mw', 'logp', 'lipinski_violations', 'binding_affinity']
    for comp in components:
        if comp in metrics_list[0]:
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
        analysis['binding_affinity_stats'] = {
            'mean': float(np.mean(binding_affinities)),
            'std': float(np.std(binding_affinities)),
            'min': float(np.min(binding_affinities)),
            'max': float(np.max(binding_affinities)),
            'best_binding': float(np.min(binding_affinities))  # Most negative = best
        }
    else:
        analysis['binding_affinity_success_rate'] = 0.0
    
    # Correlation analysis
    if len(rewards) > 5:
        qed_values = [m.get('qed', 0) for m in metrics_list]
        sas_values = [m.get('sas', 0) for m in metrics_list]
        
        analysis['correlations'] = {
            'reward_qed': float(np.corrcoef(rewards, qed_values)[0, 1]) if len(set(qed_values)) > 1 else 0,
            'reward_sas': float(np.corrcoef(rewards, sas_values)[0, 1]) if len(set(sas_values)) > 1 else 0
        }
        
        if binding_affinities and len(binding_affinities) > 5:
            # Only compute correlation for molecules with binding affinity
            binding_indices = [i for i, m in enumerate(metrics_list) if m.get('binding_affinity') is not None]
            binding_rewards = [rewards[i] for i in binding_indices]
            
            if len(binding_rewards) > 1:
                analysis['correlations']['reward_binding'] = float(
                    np.corrcoef(binding_rewards, binding_affinities)[0, 1]
                )
    
    return analysis