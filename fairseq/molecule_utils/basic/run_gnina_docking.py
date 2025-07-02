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
    exhaustiveness: int = 8,
    num_modes: int = 9,
    device: int = 0,
    cnn_scoring: str = "rescore",
    scoring: str = "vina",
) -> Optional[float]:
    """
    Performs molecular docking of a ligand to a target protein using GNINA.

    This function takes a PDB ID and a ligand SMILES string, prepares the necessary
    input files, runs the GNINA docking tool, and returns the best binding affinity
    score. It includes logic for caching results and automatically determining
    the docking box unless one is explicitly provided.
    
    Args:
        pdb_id: The PDB identifier for the target protein.
        ligand_smiles: The SMILES string of the ligand to be docked.
        pdb_path: Optional path to a user-provided PDB file.
        output_complex_path: Optional path to save the output receptor-ligand complex.
        gnina_bin_path: Path to the GNINA executable.
        split_cache_path: Path to the directory for caching split PDB files.
        pdb_cache_path: Path to the directory for caching PDB files.
        ccd_cache_path: Path to the directory for caching CCD files.
        docking_result_cache: A mutable mapping to be used as a cache for docking results.
        box_center: The explicit center coordinates (x, y, z) for the docking box.
        box_size: The dimensions (x, y, z) of the docking box in Angstroms.
        exhaustiveness: The exhaustiveness level for the GNINA search.
        num_modes: The number of binding modes to generate.
        device: The ID of the GPU device to use for GNINA.
        cnn_scoring: The CNN scoring mode to use (e.g., 'rescore', 'refinement').
        scoring: The scoring function to use (e.g., 'vina', 'vinardo').
        
    Returns:
        The best binding energy in kcal/mol, or None if docking fails.
    """
    pdb_id = pdb_id.lower()
    if pdb_path is not None:
        docking_result_cache = None
        raise NotImplementedError('pdb_path is not implemented now.')

    if docking_result_cache is not None:
        affinity = docking_result_cache.get((pdb_id, ligand_smiles), _DOCKING_CACHE_SENTINEL)
        if affinity is _DOCKING_CACHE_SENTINEL:
            affinity = docking_result_cache.get((pdb_id, ligand_smiles, box_center), _DOCKING_CACHE_SENTINEL)
        if affinity is not _DOCKING_CACHE_SENTINEL:
            logging.info('📦 Retrieved GNINA docking result from cache')
            return affinity

    try:
        gnina = GNINA(
            binary_path=gnina_bin_path,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            device=device,
            cnn_scoring=cnn_scoring,
            scoring=scoring,
        )
    except RuntimeError as e:
        logging.error(f"Failed to initialize GNINA: {e}")
        return None

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

    candidate_affinities = []

    if box_center is not None:
        if box_size is None:
            box_size = (20., 20., 20.)
            
        logging.info(f'🎯 Running GNINA docking with specified box center {box_center}')
        
        try:
            affinity = gnina.query_box(
                receptor_path=receptor_filename,
                ligand_path=ligand_path,
                center=box_center,
                box=box_size,
                output_complex_path=output_complex_path,
            )
            logging.info(f'✅ GNINA affinity at center {box_center}: {affinity} kcal/mol')
            candidate_affinities.append(affinity)
        except GNINAError as e:
            logging.warning(f"❌ GNINA failed for box center {box_center}: {e}")
    else:
        autobox_filenames = split_result.ligand_filenames.copy()
        if not autobox_filenames:
            logging.warning("⚠️ No autobox ligands found. Using receptor for autobox.")
            autobox_filenames.append(None)

        logging.info(f"🚀 Running GNINA docking on {len(autobox_filenames)} autobox candidates")
        
        for i, autobox_filename in enumerate(autobox_filenames):
            try:
                affinity = gnina.query(
                    receptor_path=receptor_filename,
                    ligand_path=ligand_path,
                    autobox_ligand_path=autobox_filename,
                    output_complex_path=output_complex_path,
                )
                logging.info(f'✅ GNINA affinity for autobox {i+1}/{len(autobox_filenames)}: {affinity} kcal/mol')
                candidate_affinities.append(affinity)
            except GNINAError as e:
                logging.warning(f"❌ GNINA failed for autobox {autobox_filename}: {e}")
                continue

    if not candidate_affinities:
        logging.warning(f"⚠️ No successful GNINA docking scores for {pdb_id}")
        return None

    affinity = min(candidate_affinities)
    logging.info(f'🏆 Best GNINA affinity for {pdb_id}: {affinity} kcal/mol')

    if docking_result_cache is not None:
        docking_result_cache[(pdb_id, ligand_smiles, box_center)] = affinity

    return affinity