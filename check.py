import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, AllChem

def validate_smiles(smiles_list):
    valid_smiles = []
    invalid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_indices.append(i)
        else:
            valid_smiles.append(smi)
    return valid_smiles, invalid_indices


def compute_metrics(smiles_list):
    qeds, logps = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            qeds.append(QED.qed(mol))
            logps.append(Descriptors.MolLogP(mol))
    return np.array(qeds), np.array(logps)


def compute_diversity(smiles_list):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2)
           for smi in smiles_list if Chem.MolFromSmiles(smi)]
    n = len(fps)
    if n < 2:
        return 0.0, 0.0
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for i in range(n) for j in range(i+1, n)]
    return np.mean(sims), np.std(sims)


def summarize(smiles_list):
    valid_smiles, invalid_indices = validate_smiles(smiles_list)
    qeds, logps = compute_metrics(valid_smiles)
    diversity_mean, diversity_std = compute_diversity(valid_smiles)
    
    summary = {
        "valid_count": len(valid_smiles),
        "invalid_count": len(invalid_indices),
        "validity_pct": 100 * len(valid_smiles) / len(smiles_list),
        "mean_qed": float(np.mean(qeds)) if qeds.size > 0 else 0.0,
        "mean_logp": float(np.mean(logps)) if logps.size > 0 else 0.0,
        "diversity_mean": diversity_mean,
        "diversity_std": diversity_std
    }
    return summary
