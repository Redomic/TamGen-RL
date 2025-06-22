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
    """Docking for one PDB-ID and ligand SMILES using GNINA.
    
    Args:
        pdb_id: PDB identifier for the target protein.
        ligand_smiles: SMILES string of the ligand to dock.
        pdb_path: (Optional) user provided PDB file instead of PDB ID.
        output_complex_path: (Optional) output receptor-ligand complex path.
        gnina_bin_path: Path to GNINA binary.
        split_cache_path: Path to split PDB cache directory.
        pdb_cache_path: Path to PDB cache directory.
        ccd_cache_path: Path to CCD cache directory.
        docking_result_cache: Cache for storing docking results.
        box_center: Center coordinates (x, y, z) for docking box.
        box_size: Box dimensions (x, y, z) in Angstroms.
        exhaustiveness: GNINA exhaustiveness parameter (higher = more thorough).
        num_modes: Number of binding modes to generate.
        device: GPU device ID for GNINA.
        cnn_scoring: CNN scoring mode (none, rescore, refinement, all).
        scoring: Scoring function (ad4_scoring, dkoes_fast, vina, vinardo).
        
    Returns:
        Best binding energy in kcal/mol, or None if docking failed.
    """
    pdb_id = pdb_id.lower()
    if pdb_path is not None:
        docking_result_cache = None
        raise NotImplementedError('pdb_path is not implemented now.')

    # Check cache first
    if docking_result_cache is not None:
        affinity = docking_result_cache.get((pdb_id, ligand_smiles), _DOCKING_CACHE_SENTINEL)
        if affinity is _DOCKING_CACHE_SENTINEL:
            affinity = docking_result_cache.get((pdb_id, ligand_smiles, box_center), _DOCKING_CACHE_SENTINEL)
        if affinity is not _DOCKING_CACHE_SENTINEL:
            logging.info('📦 Retrieved GNINA docking result from cache')
            return affinity

    # Initialize GNINA
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

    # Set default cache paths
    if split_cache_path is None:
        split_cache_path = config.split_pdb_cache_path()
    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()
    if ccd_cache_path is None:
        ccd_cache_path = config.pdb_ccd_path()

    # Split PDB complex to get receptor and ligands
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

    # Convert SMILES to PDB
    try:
        ligand_pdb_str = smi2pdb(ligand_smiles, compute_coord=True, optimize='UFF')
    except ValueError as e:
        logging.warning(f"⚠️ Ligand conversion failed: {e}")
        return None

    # Save ligand PDB to temporary file
    ligand_dir = Path("debug_docking_failures")
    ligand_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = ligand_dir / f"{pdb_id}_{uuid.uuid4().hex[:8]}.pdb"
    ligand_path.write_text(ligand_pdb_str)
    logging.info(f"🧪 Saved ligand PDB to: {ligand_path} (SMILES: {ligand_smiles})")

    candidate_affinities = []

    # Run docking with specified box center
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
        # Run docking with autobox ligands
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

    # Select best affinity (most negative for binding energy)
    if not candidate_affinities:
        logging.warning(f"⚠️ No successful GNINA docking scores for {pdb_id}")
        return None

    affinity = min(candidate_affinities)
    logging.info(f'🏆 Best GNINA affinity for {pdb_id}: {affinity} kcal/mol')

    # Cache result
    if docking_result_cache is not None:
        docking_result_cache[(pdb_id, ligand_smiles, box_center)] = affinity

    return affinity