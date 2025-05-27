import os
from glob import glob

from rdkit import Chem

def prepare_pdb_data(pdb_id, ligand_inchi=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return
    
    os.system(f"mkdir -p {DemoDataFolder}")
    if ligand_inchi is None:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id", file=fw)
            print(f"{pdb_id}", file=fw)
    else:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id,ligand_inchi", file=fw)
            print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def prepare_pdb_data_center(pdb_id, scaffold_file=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return

    with open("tmp_pdb.csv", "w") as fw:
        print("pdb_id,ligand_inchi", file=fw)
        print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def filter_generated_cmpd(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sssr = Chem.GetSymmSSSR(m)
    if len(sssr) <= 1:
        return None
    if len(sssr) >= 4:
        return None
    if smi.lower().count('p') > 3:
        return None
    s = Chem.MolToSmiles(m)
    return s, m