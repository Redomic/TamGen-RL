# reward_utils.py

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Cache to avoid recomputation of scaffold penalties
scaffold_cache = {}

def compute_advanced_reward(smiles, dock_score=None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        qed = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)

        lipinski = int((mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10) and (rot_bonds <= 10))

        reward = qed + 0.5 * lipinski
        if dock_score is not None:
            reward += -0.01 * dock_score

        return reward
    except Exception:
        return 0.0

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def analyze_reward_distribution(smiles_list):
    rewards = []
    unique_scaffolds = set()

    for smiles in smiles_list:
        r = compute_advanced_reward(smiles)
        rewards.append(r)
        scaffold = get_scaffold(smiles)
        if scaffold:
            unique_scaffolds.add(scaffold)

    return np.mean(rewards), len(unique_scaffolds)

def compute_diversity_bonus(smiles_list, weight=0.1):
    bonus = 0.0
    seen = set()
    for smiles in smiles_list:
        scaffold = get_scaffold(smiles)
        if scaffold and scaffold not in seen:
            seen.add(scaffold)
            bonus += weight
    return bonus
