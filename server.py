import os
import glob
import pandas as pd
from TamGen_RL import TamGenRL
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from TamGen_Demo import TamGenDemo, prepare_pdb_data
from db import db
from pydantic import BaseModel
import uuid
from datetime import datetime
from types import MethodType

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FastAPI app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PDBRequest model
class PDBRequest(BaseModel):
    pdb_id: str

# InferenceConfig model
class InferenceConfig(BaseModel):
    campaign_id: str
    pdb_id: str
    m_sample: int = 10000
    num_iter: int = 30
    alpha: float = 0.2
    top_k: int = 20
    lambda_sas: float = 0.3
    lambda_logp: float = 0.1
    lambda_mw: int = 0.1
    lambda_qed: float = 0.2
    lambda_docking: float = 1.0
    maxseed: int = 20

class RunCampaignRequest(BaseModel):
    pdb_id: str
    # you can extend with sampling params later

# Endpoint to prepare PDB
@app.post("/prepare_pdb")
async def prepare_pdb(request: PDBRequest):
    try:
        print("Preparing")
        pdb_id = request.pdb_id
        prepare_pdb_data(pdb_id)
        worker = TamGenDemo()
        worker.reload_data(subset="gen_" + pdb_id.lower())
        return {"success": True, "message": f"PDB {pdb_id} prepared and loaded."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/run_campaign")
async def run_campaign(req: RunCampaignRequest):
    if not req.pdb_id:
        raise HTTPException(status_code=400, detail="pdb_id is required")
    # Ensure campaigns collection exists
    if not db.has_collection("campaigns"):
        db.create_collection("campaigns")
    # 1. Create campaign
    campaign_id = uuid.uuid4().hex
    campaigns_col = db.collection("campaigns")
    campaigns_col.insert({
        "_key": campaign_id,
        "pdb_id": req.pdb_id,
        "active": True,
        "created_at": datetime.utcnow().isoformat(),
        "final_smiles": None
    })
    # Ensure iterations collection exists
    if not db.has_collection("iterations"):
        db.create_collection("iterations")
    iterations_col = db.collection("iterations")
    # Ensure edge collection exists
    if not db.has_collection("campaign-iterations"):
        db.create_collection("campaign-iterations", edge=True)
    edges_col = db.collection("campaign-iterations")

    worker = TamGenRL(
        data="./TamGen_Demo_Data",
        ckpt="checkpoints/crossdocked_model/checkpoint_best.pt",
        use_conditional=True
    )
    worker.reload_data(subset="gen_" + req.pdb_id.lower())
    # 4. Monkey-patch save to also write to DB
    original_save = worker._save_iteration_results
    def save_and_insert(self, iteration, smiles_list, rewards, detailed_results, z_vectors=None):
        original_save(iteration, smiles_list, rewards, detailed_results, z_vectors)
        # build records
        records = []
        for smi, reward, metric in zip(smiles_list, rewards, detailed_results):
            rec = {
                "SMILES": smi,
                "Reward": reward,
                "QED": metric.get("qed"),
                "SAS": metric.get("sas"),
                "MW": metric.get("mw"),
                "LogP": metric.get("logp"),
            }
            records.append(rec)
        # Insert into iterations collection
        meta = iterations_col.insert({
            "campaign_id": campaign_id,
            "iteration": iteration,
            "results": records
        })
        # Create edge from campaign to iteration
        edges_col.insert({
            "_from": f"campaigns/{campaign_id}",
            "_to": f"iterations/{meta['_key']}"
        })
    worker._save_iteration_results = MethodType(save_and_insert, worker)
    
    # 5. Run sampling synchronously
    # final_smiles = worker.sample(
    #     m_sample=10000,
    #     num_iter=20,
    #     use_cuda=torch.cuda.is_available(),
    #     save_intermediates=True
    # )

    final_smiles = worker.sample(
        m_sample=10000,      
        num_iter=20,     
        latent_dim=256,
        alpha=0.4,
        top_k=10,
        lambda_sas=0.3,
        lambda_logp=0.1,
        lambda_mw=0.1,
        maxseed=20,       
        use_cuda=torch.cuda.is_available()
    )

    campaigns_col.update({
        "_key": campaign_id,
        "active": False,
        "generated_smiles": final_smiles
    })
    return {"campaign_id": campaign_id, "final_smiles": final_smiles}