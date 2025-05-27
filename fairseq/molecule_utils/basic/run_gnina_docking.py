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

    import uuid
    try:
        ligand_pdb_str = smi2pdb(ligand_smiles, compute_coord=True, optimize='UFF')
    except ValueError as e:
        logging.warning(f"⚠️ Ligand conversion failed: {e}")
        return None

    ligand_dir = Path("debug_docking_failures")
    ligand_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = ligand_dir / f"{pdb_id}_{uuid.uuid4().hex[:8]}.pdb"
    ligand_path.write_text(ligand_pdb_str)
    logging.info(f"🧪 Saved ligand PDB to: {ligand_path} (SMILES: {ligand_smiles})")

    autobox_filenames = split_result.ligand_filenames.copy()
    if not autobox_filenames:
        logging.warning("⚠️ No autobox ligands found. Proceeding without autobox.")
        autobox_filenames.append(None)

    logging.info(f"🚀 Running GNINA batch docking on {len(autobox_filenames)} autobox candidates.")
    candidate_affinities = []
    # For each autobox ligand, run docking
    for autobox_filename in autobox_filenames:
        gnina.exhaustiveness = 1
        gnina.num_modes = 1
        gnina.cnn_scoring = "cnnscore_only"
        gnina.device = 0
        try:
            affinity = gnina.query(
                receptor_path=receptor_filename,
                ligand_path=ligand_path,
                autobox_ligand_path=autobox_filename,
                output_complex_path=output_complex_path or Path('/dev/null'),
            )
        except GNINAError as e:
            logging.warning(
                f"❌ GNINA batch failed for target={receptor_filename}, ligand={ligand_path}, autobox={autobox_filename}.\n{e}"
            )
            continue
        candidate_affinities.append(affinity)

    if not candidate_affinities:
        logging.warning(f"⚠️ No successful docking scores for {pdb_id}")
        return None

    affinity = min(candidate_affinities)
    if docking_result_cache is not None:
        docking_result_cache[(pdb_id, ligand_smiles, box_center)] = affinity
    return affinity