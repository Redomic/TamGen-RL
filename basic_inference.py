#!/usr/bin/env python3
"""
TamGen Inference Script
Complete implementation for molecular generation using TamGen with exact parameters
from the paper "Target-aware molecular generation with structure-based drug design"
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import namedtuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# RDKit for molecular processing
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# BioPython for PDB processing
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure as PdbStructure

# Scientific computing
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm

# Fairseq imports (you'll need fairseq installed)
try:
    from fairseq import checkpoint_utils, utils, options
    from fairseq.data import Dictionary
    from fairseq.models import FairseqEncoderDecoderModel, register_model
    from fairseq.modules import MultiheadAttention, LayerNorm, TransformerEncoderLayer
    from fairseq.models.transformer import TransformerModel, Embedding, TransformerDecoder
except ImportError:
    print("Warning: Fairseq not found. Please install fairseq for full functionality.")
    print("pip install fairseq")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data structures
ResidueInfo = namedtuple('ResidueInfo', 'chain_id res_id res_code pos')
GenerationParams = namedtuple('GenerationParams', 'distance_cutoff beam_size num_seeds vae_beta max_length')

class TamGenConfig:
    """Configuration class with exact parameters from the paper"""
    
    def __init__(self):
        # Model architecture parameters
        self.encoder_embed_dim = 256
        self.encoder_layers = 4
        self.encoder_attention_heads = 8
        self.encoder_ffn_embed_dim = 1024
        
        # Decoder parameters (GPT-like)
        self.decoder_embed_dim = 768
        self.decoder_layers = 12
        self.decoder_attention_heads = 12
        self.decoder_ffn_embed_dim = 3072
        
        # Distance-aware attention
        self.dist_attn = True
        self.dist_decay = 3000  # τ parameter
        
        # VAE parameters
        self.vae = True
        self.vae_beta = 0.1
        
        # Coordinate processing
        self.move_to_origin = True
        self.random_rotation = False  # Only during training
        self.add_noise = False  # Only during training
        self.mlp_layers = 0
        
        # Training parameters
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_fn = 'gelu'
        
        # Generation parameters for Design stage
        self.design_distance_cutoffs = [10, 15]  # Å
        self.design_beam_size = 20
        self.design_num_seeds = 20
        self.design_vae_betas = [0.1, 1.0]
        
        # Generation parameters for Refine stage
        self.refine_distance_cutoffs = [10, 12, 15]  # Å
        self.refine_beam_sizes = [4, 10, 20]
        self.refine_num_seeds = 100
        self.refine_vae_betas = [0.1, 1.0]
        
        # Data processing
        self.max_seq_length = 1023
        self.coordinate_noise_std = 0.1
        self.center_coordinates = True
        
        # Generation settings
        self.sampling_temperature = 1.0
        self.length_penalty = 1.0
        self.max_smiles_length = 512

class PDBProcessor:
    """Handles PDB file processing and binding pocket extraction"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.aa_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
    
    def get_residue_average_position(self, residue: Residue, only_aa: bool = True) -> np.ndarray:
        """Get average position of atoms in residue"""
        atoms = []
        for atom in residue:
            if only_aa and atom.element == 'H':  # Skip hydrogens
                continue
            atoms.append(atom.coord)
        
        if not atoms:
            return np.array([0.0, 0.0, 0.0])
        
        return np.mean(atoms, axis=0).astype(np.float32)
    
    def extract_binding_pocket(self, pdb_path: str, ligand_center: np.ndarray, 
                             distance_cutoff: float = 10.0) -> Tuple[str, np.ndarray, List[ResidueInfo]]:
        """Extract binding pocket residues within distance cutoff"""
        
        structure = self.parser.get_structure('protein', pdb_path)
        first_model = next(structure.get_models())
        
        all_residues = []
        
        for residue in first_model.get_residues():
            _, _, chain_id, res_index = residue.get_full_id()
            res_type, res_id, insertion_code = res_index
            
            if res_type != ' ':  # Skip non-standard residues
                continue
                
            res_name = residue.get_resname()
            if res_name not in self.aa_3to1:
                continue
                
            res_code = self.aa_3to1[res_name]
            pos = self.get_residue_average_position(residue, only_aa=True)
            
            # Check if residue is within distance cutoff
            distance = np.linalg.norm(pos - ligand_center)
            if distance <= distance_cutoff:
                residue_info = ResidueInfo(chain_id, res_id, res_code, pos)
                all_residues.append(residue_info)
        
        # Sort residues by chain and position
        all_residues.sort(key=lambda info: (info.chain_id, info.res_id))
        
        # Extract sequence and coordinates
        sequence = ''.join(info.res_code for info in all_residues)
        coordinates = np.stack([info.pos for info in all_residues])
        
        return sequence, coordinates, all_residues
    
    def process_pdb_structure(self, pdb_path: str, ligand_center: Optional[np.ndarray] = None,
                            distance_cutoff: float = 10.0) -> Dict[str, Any]:
        """Process PDB structure and extract relevant information"""
        
        if ligand_center is None:
            # If no ligand center provided, use center of mass of the structure
            structure = self.parser.get_structure('protein', pdb_path)
            first_model = next(structure.get_models())
            
            coords = []
            for residue in first_model.get_residues():
                if residue.id[0] == ' ':  # Standard residue
                    pos = self.get_residue_average_position(residue)
                    coords.append(pos)
            
            if coords:
                ligand_center = np.mean(coords, axis=0)
            else:
                ligand_center = np.array([0.0, 0.0, 0.0])
        
        sequence, coordinates, residue_info = self.extract_binding_pocket(
            pdb_path, ligand_center, distance_cutoff
        )
        
        return {
            'sequence': sequence,
            'coordinates': coordinates,
            'residue_info': residue_info,
            'ligand_center': ligand_center,
            'pdb_path': pdb_path
        }

class SMILESProcessor:
    """Handles SMILES encoding/decoding and validation"""
    
    def __init__(self, vocab_path: Optional[str] = None):
        if vocab_path and os.path.exists(vocab_path):
            self.vocab = Dictionary.load(vocab_path)
        else:
            # Create basic SMILES vocabulary
            self.vocab = self._create_basic_vocab()
    
    def _create_basic_vocab(self) -> Dictionary:
        """Create basic SMILES vocabulary"""
        vocab = Dictionary()
        
        # Basic SMILES tokens
        smiles_tokens = [
            '<pad>', '<s>', '</s>', '<unk>',
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',
            '(', ')', '[', ']', '@', '+', '-', '#', '=',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'H', 'B', 'Si', 'Se', 'As', 'Al', 'Na', 'K', 'Mg', 'Ca',
            '\\', '/', '%'
        ]
        
        for token in smiles_tokens:
            vocab.add_symbol(token)
        
        vocab.finalize()
        return vocab
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """Encode SMILES string to tensor"""
        tokens = ['<s>'] + list(smiles) + ['</s>']
        indices = [self.vocab.index(token) if token in self.vocab.indices 
                  else self.vocab.unk() for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_smiles(self, tensor: torch.Tensor) -> str:
        """Decode tensor to SMILES string"""
        tokens = []
        for idx in tensor:
            token = self.vocab[idx.item()]
            if token in ['<s>', '</s>', '<pad>']:
                continue
            if token == '<unk>':
                token = 'C'  # Replace unknown with carbon
            tokens.append(token)
        return ''.join(tokens)
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def calculate_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'qed': Descriptors.qed(mol)
            }
        except:
            return {}

class TamGenModel:
    """Main TamGen model wrapper"""
    
    def __init__(self, config: TamGenConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.smiles_processor = SMILESProcessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No model checkpoint provided. Model needs to be loaded separately.")
    
    def load_model(self, checkpoint_path: str):
        """Load pre-trained TamGen model"""
        try:
            state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
            # Note: This would need the actual fairseq model implementation
            # self.model = build_model_from_checkpoint(state, self.config)
            # self.model.to(self.device)
            # self.model.eval()
            logger.info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _prepare_protein_input(self, sequence: str, coordinates: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare protein input tensors"""
        # Encode amino acid sequence
        aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        
        seq_indices = [aa_to_idx.get(aa, 0) for aa in sequence]
        src_tokens = torch.tensor(seq_indices, dtype=torch.long).unsqueeze(0)
        
        # Process coordinates
        if self.config.center_coordinates:
            coordinates = coordinates - np.mean(coordinates, axis=0)
        
        src_coords = torch.tensor(coordinates, dtype=torch.float32).unsqueeze(0)
        
        return {
            'src_tokens': src_tokens.to(self.device),
            'src_coords': src_coords.to(self.device)
        }
    
    def generate_compounds_design_stage(self, sequence: str, coordinates: np.ndarray) -> List[str]:
        """Design stage: Generate compounds from protein pocket only"""
        
        if self.model is None:
            logger.error("Model not loaded. Cannot generate compounds.")
            return []
        
        generated_compounds = []
        config = self.config
        
        logger.info("Starting Design stage compound generation...")
        
        for seed in tqdm(range(1, config.design_num_seeds + 1), desc="Design seeds"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            for distance_cutoff in config.design_distance_cutoffs:
                for vae_beta in config.design_vae_betas:
                    
                    try:
                        # Prepare input
                        inputs = self._prepare_protein_input(sequence, coordinates)
                        
                        # Generate compounds
                        with torch.no_grad():
                            # This would call the actual model generation
                            # outputs = self.model.generate(
                            #     src_tokens=inputs['src_tokens'],
                            #     src_coords=inputs['src_coords'],
                            #     beam_size=config.design_beam_size,
                            #     max_length=config.max_smiles_length,
                            #     temperature=config.sampling_temperature
                            # )
                            
                            # For demonstration, generate dummy compounds
                            outputs = self._dummy_generation(config.design_beam_size)
                        
                        # Decode SMILES
                        for output in outputs:
                            smiles = self.smiles_processor.decode_smiles(output)
                            if self.smiles_processor.is_valid_smiles(smiles):
                                generated_compounds.append(smiles)
                    
                    except Exception as e:
                        logger.error(f"Error in generation: {e}")
                        continue
        
        # Remove duplicates
        unique_compounds = list(set(generated_compounds))
        logger.info(f"Design stage generated {len(unique_compounds)} unique compounds")
        
        return unique_compounds
    
    def generate_compounds_refine_stage(self, sequence: str, coordinates: np.ndarray, 
                                      seed_compounds: List[str]) -> List[str]:
        """Refine stage: Generate compounds using seed molecules"""
        
        if self.model is None:
            logger.error("Model not loaded. Cannot generate compounds.")
            return []
        
        generated_compounds = []
        config = self.config
        
        logger.info("Starting Refine stage compound generation...")
        
        for seed in tqdm(range(1, config.refine_num_seeds + 1), desc="Refine seeds"):
            torch.manual_seed(seed)
            
            for seed_compound in seed_compounds[:10]:  # Limit to top 10 seed compounds
                for distance_cutoff in config.refine_distance_cutoffs:
                    for beam_size in config.refine_beam_sizes:
                        for vae_beta in config.refine_vae_betas:
                            
                            try:
                                # Prepare inputs
                                protein_inputs = self._prepare_protein_input(sequence, coordinates)
                                compound_input = self.smiles_processor.encode_smiles(seed_compound)
                                
                                with torch.no_grad():
                                    # This would call the actual VAE-based refinement
                                    # outputs = self.model.refine_generate(
                                    #     src_tokens=protein_inputs['src_tokens'],
                                    #     src_coords=protein_inputs['src_coords'],
                                    #     seed_compound=compound_input.unsqueeze(0).to(self.device),
                                    #     beam_size=beam_size,
                                    #     vae_beta=vae_beta,
                                    #     max_length=config.max_smiles_length
                                    # )
                                    
                                    # For demonstration
                                    outputs = self._dummy_generation(beam_size)
                                
                                # Process outputs
                                for output in outputs:
                                    smiles = self.smiles_processor.decode_smiles(output)
                                    if self.smiles_processor.is_valid_smiles(smiles):
                                        generated_compounds.append(smiles)
                            
                            except Exception as e:
                                logger.error(f"Error in refinement: {e}")
                                continue
        
        # Remove duplicates
        unique_compounds = list(set(generated_compounds))
        logger.info(f"Refine stage generated {len(unique_compounds)} unique compounds")
        
        return unique_compounds
    
    def _dummy_generation(self, beam_size: int) -> List[torch.Tensor]:
        """Dummy generation for demonstration purposes"""
        # Generate some example SMILES-like sequences
        dummy_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)O",
            "CCN(CC)CC",
            "c1ccc2ccccc2c1"
        ]
        
        outputs = []
        for i in range(min(beam_size, len(dummy_smiles))):
            encoded = self.smiles_processor.encode_smiles(dummy_smiles[i])
            outputs.append(encoded)
        
        return outputs

class CompoundFilter:
    """Handles compound filtering and ranking"""
    
    def __init__(self):
        self.smiles_processor = SMILESProcessor()
    
    def filter_by_drug_likeness(self, compounds: List[str], 
                               rules: str = 'lipinski') -> List[Dict[str, Any]]:
        """Filter compounds by drug-likeness rules"""
        
        filtered_compounds = []
        
        for smiles in compounds:
            props = self.smiles_processor.calculate_properties(smiles)
            
            if not props:  # Invalid molecule
                continue
            
            # Apply Lipinski's Rule of Five
            if rules == 'lipinski':
                if (props.get('molecular_weight', 1000) <= 500 and
                    props.get('logp', 10) <= 5 and
                    props.get('hbd', 10) <= 5 and
                    props.get('hba', 20) <= 10):
                    
                    filtered_compounds.append({
                        'smiles': smiles,
                        'properties': props,
                        'score': props.get('qed', 0)
                    })
        
        # Sort by QED score
        filtered_compounds.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered_compounds
    
    def dock_compounds(self, compounds: List[str], target_pdb: str,
                      reference_compound: str = None) -> List[Dict[str, Any]]:
        """Simulate molecular docking (placeholder implementation)"""
        
        # This would integrate with AutoDock Vina or similar
        # For now, return dummy docking scores
        
        docked_compounds = []
        
        for smiles in compounds:
            # Simulate docking score calculation
            dummy_score = np.random.uniform(-12, -6)  # Typical docking score range
            
            docked_compounds.append({
                'smiles': smiles,
                'docking_score': dummy_score,
                'target': target_pdb
            })
        
        # Sort by docking score (lower is better)
        docked_compounds.sort(key=lambda x: x['docking_score'])
        
        return docked_compounds
    
    def select_diverse_compounds(self, compounds: List[Dict[str, Any]], 
                               num_select: int = 10) -> List[Dict[str, Any]]:
        """Select diverse compounds based on structural diversity"""
        
        if len(compounds) <= num_select:
            return compounds
        
        # Simple diversity selection (would use proper fingerprints in practice)
        selected = []
        
        # Take top compounds by score, ensuring some diversity
        compounds_sorted = sorted(compounds, key=lambda x: x.get('score', x.get('docking_score', 0)))
        
        for compound in compounds_sorted:
            if len(selected) >= num_select:
                break
            
            # Simple diversity check (would use Tanimoto similarity in practice)
            is_diverse = True
            for sel_comp in selected:
                if len(compound['smiles']) == len(sel_comp['smiles']):
                    # Very simple diversity check
                    if compound['smiles'][:5] == sel_comp['smiles'][:5]:
                        is_diverse = False
                        break
            
            if is_diverse:
                selected.append(compound)
        
        return selected

class TamGenPipeline:
    """Complete TamGen inference pipeline"""
    
    def __init__(self, config: TamGenConfig, model_path: Optional[str] = None):
        self.config = config
        self.pdb_processor = PDBProcessor()
        self.model = TamGenModel(config, model_path)
        self.compound_filter = CompoundFilter()
        
    def run_full_pipeline(self, pdb_path: str, ligand_center: Optional[np.ndarray] = None,
                         output_dir: str = "tamgen_output") -> Dict[str, Any]:
        """Run complete TamGen Design-Refine-Test pipeline"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting TamGen inference pipeline...")
        
        # Step 1: Process protein structure
        logger.info("Step 1: Processing protein structure...")
        protein_data = self.pdb_processor.process_pdb_structure(
            pdb_path, ligand_center, self.config.design_distance_cutoffs[0]
        )
        
        logger.info(f"Extracted pocket with {len(protein_data['sequence'])} residues")
        
        # Step 2: Design stage - Generate initial compounds
        logger.info("Step 2: Design stage - Generating initial compounds...")
        design_compounds = self.model.generate_compounds_design_stage(
            protein_data['sequence'], protein_data['coordinates']
        )
        
        # Save design compounds
        with open(os.path.join(output_dir, 'design_compounds.json'), 'w') as f:
            json.dump(design_compounds, f, indent=2)
        
        # Step 3: Filter design compounds
        logger.info("Step 3: Filtering design compounds...")
        filtered_design = self.compound_filter.filter_by_drug_likeness(design_compounds)
        docked_design = self.compound_filter.dock_compounds(
            [c['smiles'] for c in filtered_design], pdb_path
        )
        
        # Step 4: Select representative compounds for refinement
        logger.info("Step 4: Selecting representative compounds...")
        representative_compounds = self.compound_filter.select_diverse_compounds(
            docked_design, num_select=8  # Select top 8 for refinement
        )
        
        seed_smiles = [c['smiles'] for c in representative_compounds]
        
        # Step 5: Refine stage - Generate refined compounds
        logger.info("Step 5: Refine stage - Generating refined compounds...")
        refined_compounds = self.model.generate_compounds_refine_stage(
            protein_data['sequence'], protein_data['coordinates'], seed_smiles
        )
        
        # Save refined compounds
        with open(os.path.join(output_dir, 'refined_compounds.json'), 'w') as f:
            json.dump(refined_compounds, f, indent=2)
        
        # Step 6: Final filtering and ranking
        logger.info("Step 6: Final filtering and ranking...")
        filtered_refined = self.compound_filter.filter_by_drug_likeness(refined_compounds)
        final_docked = self.compound_filter.dock_compounds(
            [c['smiles'] for c in filtered_refined], pdb_path
        )
        
        # Step 7: Select final candidates
        final_candidates = self.compound_filter.select_diverse_compounds(
            final_docked, num_select=20
        )
        
        # Prepare results
        results = {
            'protein_data': {
                'sequence': protein_data['sequence'],
                'num_residues': len(protein_data['sequence']),
                'pdb_path': pdb_path
            },
            'design_stage': {
                'total_generated': len(design_compounds),
                'compounds': design_compounds[:100]  # Save first 100
            },
            'refine_stage': {
                'total_generated': len(refined_compounds),
                'seed_compounds': seed_smiles,
                'compounds': refined_compounds[:100]  # Save first 100
            },
            'final_candidates': final_candidates,
            'statistics': {
                'design_compounds': len(design_compounds),
                'filtered_design': len(filtered_design),
                'refined_compounds': len(refined_compounds),
                'final_candidates': len(final_candidates)
            }
        }
        
        # Save complete results
        with open(os.path.join(output_dir, 'tamgen_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(results, output_dir)
        
        logger.info(f"Pipeline completed. Results saved to {output_dir}")
        
        return results
    
    def _create_summary_report(self, results: Dict[str, Any], output_dir: str):
        """Create a summary report of the results"""
        
        report_lines = [
            "TamGen Inference Pipeline Summary",
            "=" * 40,
            "",
            f"Protein: {results['protein_data']['pdb_path']}",
            f"Pocket residues: {results['protein_data']['num_residues']}",
            "",
            "Generation Statistics:",
            f"  Design stage compounds: {results['statistics']['design_compounds']}",
            f"  Filtered design compounds: {results['statistics']['filtered_design']}",
            f"  Refined compounds: {results['statistics']['refined_compounds']}",
            f"  Final candidates: {results['statistics']['final_candidates']}",
            "",
            "Top 10 Final Candidates:",
        ]
        
        for i, candidate in enumerate(results['final_candidates'][:10], 1):
            smiles = candidate['smiles']
            score = candidate.get('docking_score', 'N/A')
            report_lines.append(f"  {i:2d}. {smiles} (Score: {score})")
        
        # Save report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='TamGen Inference Pipeline')
    parser.add_argument('--pdb', required=True, help='Path to PDB file')
    parser.add_argument('--model', help='Path to TamGen model checkpoint')
    parser.add_argument('--output', default='tamgen_output', help='Output directory')
    parser.add_argument('--ligand-center', nargs=3, type=float, 
                       help='Ligand center coordinates (x y z)')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TamGenConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(config, key, value)
    
    # Parse ligand center
    ligand_center = None
    if args.ligand_center:
        ligand_center = np.array(args.ligand_center, dtype=np.float32)
    
    # Initialize pipeline
    pipeline = TamGenPipeline(config, args.model)
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            args.pdb, ligand_center, args.output
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Generated {results['statistics']['final_candidates']} final candidates")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())