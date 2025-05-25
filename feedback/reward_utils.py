from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import math
from fairseq.molecule_utils.basic.run_docking import docking

# SAS calculator setup
try:
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen
except ImportError:
    raise ImportError("RDKit must be installed with SAS score dependencies")

def compute_qed(mol):
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0

def compute_sas(mol):
    try:
        from rdkit.Chem import rdMolDescriptors
        sa_score = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
        return sa_score
    except Exception:
        return 10.0  # high = bad

def compute_reward(mol, docking_score, lambda_sas, lambda_logp, lambda_mw):
    if mol is None:
        return -10.0, {}

    qed = compute_qed(mol)
    sas = compute_sas(mol)
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    # Lipinski violations
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])

    # Nonlinear penalty
    penalty = 0.0
    if mw > 550:
        penalty += 0.5
    if logp > 5:
        penalty += 0.5
    if lipinski_violations > 2:
        penalty += 1.0

    if docking_score is None:
        raise ValueError("Docking score must be provided for reward computation.")

    docking_bonus = 0.0
    if docking_score is not None:
        docking_bonus = min(1.0, -docking_score / 10.0)  # Assumes docking score is negative

    reward = (
        qed
        - lambda_sas * sas
        - lambda_logp * abs(logp - 2.5)
        - lambda_mw * (mw / 500.0)
        - penalty
        + docking_bonus
    )

    # Log each component
    metrics = {
        "qed": qed,
        "sas": sas,
        "logp": logp,
        "mw": mw,
        "hbd": hbd,
        "hba": hba,
        "lipinski_violations": lipinski_violations,
        "penalty": penalty,
        "docking_bonus": docking_bonus,
        "reward": reward
    }

    return reward, metrics