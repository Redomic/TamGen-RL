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
    m_sample=50000,      # Tiny
    num_iter=20,      # Just 2 iterations  
    latent_dim=256,
    alpha=0.4,
    top_k=10,
    lambda_sas=0.3,
    lambda_logp=0.1,
    lambda_mw=0.1,
    maxseed=10,       # Just 2 seeds
    use_cuda=torch.cuda.is_available()
)

# === Save or Analyze Results ===
print(f"\nFinal set of SMILES ({len(final_smiles)} molecules):")
for smi in final_smiles:
    print(smi)
