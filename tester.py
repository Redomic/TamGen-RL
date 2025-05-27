from TamGen_RL import TamGenRL
from utils import prepare_pdb_data, prepare_pdb_data_center, filter_generated_cmpd
import torch

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# === Setup TamGenRL ===
pdb_id = "3ny8"
print(f"📄 Preparing PDB: {pdb_id}")
prepare_pdb_data(pdb_id)

demo = TamGenRL(
    data="TamGen_Demo_Data",
    ckpt="checkpoints/crossdocked_model/checkpoint_best.pt",
    use_conditional=True
)
demo.reload_data(subset="gen_" + pdb_id.lower())

# === Run Closed-Loop Optimization ===
final_smiles = demo.sample(
    m_sample=500000,         # Number of molecules per iteration
    num_iter=5,           # Number of closed-loop optimization steps
    latent_dim=256,       # Latent space dimensionality (set to your model's config)
    alpha=0.5,            # Centroid shift parameter
    top_k=50,             # How many top molecules to use for shifting
    lambda_sas=0.3,       # Reward hyperparameters
    lambda_logp=0.1,
    lambda_mw=0.1,
    maxseed=20,           # Number of random seeds (first iteration)
    use_cuda=True
)

# === Save or Analyze Results ===
print(f"\nFinal set of SMILES ({len(final_smiles)} molecules):")
for smi in final_smiles:
    print(smi)
