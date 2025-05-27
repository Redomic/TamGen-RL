from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import math

# SAS calculator setup
try:
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen
except ImportError:
    raise ImportError("RDKit must be installed with SAS score dependencies")

def compute_qed(mol, device='cpu'):
    """
    Compute QED (Quantitative Estimation of Drug-likeness) for a molecule.
    Always runs on CPU by default. The 'device' argument is for future compatibility
    if torch-based or GPU-accelerated computation is added.
    Currently processes a single molecule at a time; batch support may be added in future.
    """
    # NOTE: All RDKit operations are CPU-based. If torch is used in future, use the 'device' argument.
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0

def compute_sas(mol, device='cpu'):
    """
    Compute SAS (Synthetic Accessibility Score) for a molecule.
    Always runs on CPU by default. The 'device' argument is for future compatibility
    if torch-based or GPU-accelerated computation is added.
    Currently processes a single molecule at a time; batch support may be added in future.
    """
    # NOTE: All RDKit operations are CPU-based. If torch is used in future, use the 'device' argument.
    try:
        from rdkit.Chem import rdMolDescriptors
        sa_score = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
        return sa_score
    except Exception:
        return 10.0  # high = bad

def compute_reward(mol, docking_score, lambda_sas, lambda_logp, lambda_mw, device='cpu'):
    """
    Compute the overall reward for a molecule, combining QED, SAS, logP, MW, and docking score.
    All calculations are performed on CPU by default. The 'device' argument is for future compatibility
    with torch-based or GPU-accelerated reward models.
    Currently processes a single molecule at a time; batch support may be added in future.
    To add torch support, ensure all tensors are placed on the specified device.
    """
    # NOTE: All RDKit operations are CPU-based. If torch is used in future, use the 'device' argument.
    if mol is None:
        return -10.0, {}

    qed = compute_qed(mol, device=device)
    sas = compute_sas(mol, device=device)
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

    docking_bonus = 0.0

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