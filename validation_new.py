import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, AllChem

# Validation Notebook for TamGenRL Optimization Results

def load_results_tsv(path):
    df = pd.read_csv(path, sep="\t")
    return df


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


def plot_distributions(qeds, logps, label=""):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(qeds, bins=20, kde=True, color='green')
    plt.title(f"QED Distribution {label}")
    
    plt.subplot(1, 2, 2)
    sns.histplot(logps, bins=20, kde=True, color='red')
    plt.title(f"LogP Distribution {label}")
    
    plt.tight_layout()
    plt.show()


def compute_diversity(smiles_list):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2) for smi in smiles_list]
    n = len(fps)
    if n < 2:
        return 0.0, 0.0
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    return np.mean(sims), np.std(sims)


def summarize_results(df):
    smiles = df['SMILES'].tolist()
    rewards = df['Reward'].values
    qed = df['QED'].values  # still read from file if exists; not used here though
    logp = df['LogP'].values  # same
    
    valid_smiles, invalid_idx = validate_smiles(smiles)
    unique_smiles = set(valid_smiles)
    
    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    validity_pct = 100 * len(valid_smiles) / len(smiles)
    uniqueness_pct = 100 * len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
    
    qeds_valid, logps_valid = compute_metrics(valid_smiles)
    diversity_mean, diversity_std = compute_diversity(valid_smiles)
    
    print("=== Optimization Results Summary ===")
    print(f"Total molecules:          {len(smiles)}")
    print(f"Valid molecules:          {len(valid_smiles)} ({validity_pct:.1f}%)")
    print(f"Unique molecules:         {len(unique_smiles)} ({uniqueness_pct:.1f}%)")
    print(f"Mean Reward:              {mean_reward:.4f}")
    print(f"Max Reward:               {max_reward:.4f}")
    print(f"Mean QED (valid):         {np.mean(qeds_valid):.4f}")
    print(f"Mean LogP (valid):        {np.mean(logps_valid):.4f}")
    print(f"Diversity (mean Tanimoto): {diversity_mean:.4f} ± {diversity_std:.4f}")
    print(f"Invalid SMILES count:     {len(invalid_idx)}")
    
    plot_distributions(qeds_valid, logps_valid, label="(Valid Molecules)")


# ==== Run validation on a given results file ====
start_epoch = 1
end_epoch = 42
plot_metrics_over_epochs(start_epoch, end_epoch, results_dir="latent_logs")
print_progress_deltas(start_epoch=1, end_epoch=42, results_dir="latent_logs")
epoch_to_validate = 10
validate_single_epoch(epoch_to_validate, results_dir="latent_logs")