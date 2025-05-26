"""Run docking using GNINA on a given target-ligand pair."""

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional, MutableMapping, Tuple

from .smiles_utils import smi2pdb
from .. import config
from ..database.split_complex import split_pdb_complex_paths
from ..external_tools.gnina import GNINA, GNINAError

_DOCKING_CACHE_SENTINEL = object()

def docking(
    pdb_id: str,
    ligand_smiles: str,
    *,
    pdb_path: Path = None,
    output_complex_path: Path = None,
    gnina_bin_path: Path = None,
    split_cache_path: Path = None,
    pdb_cache_path: Path = None,
    ccd_cache_path: Path = None,
    docking_result_cache: MutableMapping = None,
    box_center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = None,
) -> Optional[float]:
    """Docking for one PDB-ID and ligand SMILES using GNINA."""
    pdb_id = pdb_id.lower()
    if pdb_path is not None:
        docking_result_cache = None
        raise NotImplementedError('pdb_path is not implemented now.')

    if docking_result_cache is not None:
        affinity = docking_result_cache.get((pdb_id, ligand_smiles), _DOCKING_CACHE_SENTINEL)
        if affinity is _DOCKING_CACHE_SENTINEL:
            affinity = docking_result_cache.get((pdb_id, ligand_smiles, box_center), _DOCKING_CACHE_SENTINEL)
        if affinity is not _DOCKING_CACHE_SENTINEL:
            logging.info('📦 Retrieved docking result from cache')
            return affinity

    gnina = GNINA(binary_path=gnina_bin_path or Path("/workspace/workspace/TamGen/gnina/build/bin/gnina"))
    if not gnina.check_binary():
        raise RuntimeError('Cannot find GNINA executable.')

    if split_cache_path is None:
        split_cache_path = config.split_pdb_cache_path()
    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()
    if ccd_cache_path is None:
        ccd_cache_path = config.pdb_ccd_path()

    try:
        split_result = split_pdb_complex_paths(
            pdb_id, split_cache_path=split_cache_path,
            pdb_cache_path=pdb_cache_path, ccd_cache_path=ccd_cache_path
        )
    except RuntimeError as e:
        logging.warning(e)
        return None
    receptor_filename = split_result.target_filename
    if receptor_filename is None:
        logging.warning(f"⚠️ Cannot find target file of {pdb_id}, skipping.")
        return None

    # Batch docking logic for a single ligand (for API compatibility, use batch code for one ligand)
    import hashlib
    ligand_dir = Path("debug_docking_failures")
    ligand_dir.mkdir(parents=True, exist_ok=True)
    smiles_list = [ligand_smiles]
    ligand_pdb_list = []
    smi_to_hash = {}
    successful_conversion = True
    for smi in smiles_list:
        ligand_hash = hashlib.sha256(smi.encode()).hexdigest()[:10]
        smi_to_hash[ligand_hash] = smi
        try:
            ligand_pdb_str = smi2pdb(smi, compute_coord=True, optimize=None)
        except ValueError as e:
            logging.warning(f"⚠️ Ligand conversion failed: {e}")
            successful_conversion = False
            break
        ligand_pdb_list.append((ligand_hash, ligand_pdb_str))
    if not successful_conversion:
        return None

    # Save ligand PDBs and batch file
    for ligand_hash, ligand_pdb_str in ligand_pdb_list:
        ligand_path = ligand_dir / f"{pdb_id}_{ligand_hash}.pdb"
        if not ligand_path.exists():
            ligand_path.write_text(ligand_pdb_str)
            logging.info(f"🧪 Saved ligand PDB to: {ligand_path} (SMILES: {smi_to_hash[ligand_hash]})")
        else:
            logging.info(f"🧪 Using cached ligand PDB: {ligand_path}")

    multi_ligand_path = ligand_dir / f"{pdb_id}_batch.pdb"
    # Only cache the multi-ligand PDB if all SMILES were converted
    with open(multi_ligand_path, "w") as f:
        for ligand_hash, ligand_pdb_str in ligand_pdb_list:
            f.write(ligand_pdb_str)

    autobox_filenames = split_result.ligand_filenames.copy()
    if not autobox_filenames:
        logging.warning("⚠️ No autobox ligands found. Proceeding without autobox.")
        autobox_filenames.append(None)

    logging.info(f"🚀 Running GNINA batch docking on {len(autobox_filenames)} autobox candidates.")
    candidate_affinities = []
    # For each autobox ligand, run batch docking
    for autobox_filename in autobox_filenames:
        gnina.exhaustiveness = 1
        gnina.num_modes = 1
        gnina.cnn_scoring = "cnnscore_only"
        gnina.device = 0
        try:
            affinity_dict = gnina.batch_query(
                receptor_path=receptor_filename,
                ligand_path=multi_ligand_path,
                autobox_ligand_path=autobox_filename,
            )
        except GNINAError as e:
            logging.warning(
                f"❌ GNINA batch failed for target={receptor_filename}, multi_ligand={multi_ligand_path}, autobox={autobox_filename}.\n{e}"
            )
            continue
        # Map results back to SMILES using hash
        docking_scores = [
            affinity_dict.get(hashlib.sha256(s.encode()).hexdigest()[:10], None) for s in smiles_list
        ]
        # For single ligand, just append its score if successful
        if docking_scores[0] is not None:
            candidate_affinities.append(docking_scores[0])

    if not candidate_affinities:
        logging.warning(f"⚠️ No successful docking scores for {pdb_id}")
        return None

    affinity = min(candidate_affinities)
    if docking_result_cache is not None:
        docking_result_cache[(pdb_id, ligand_smiles, box_center)] = affinity

    return affinity