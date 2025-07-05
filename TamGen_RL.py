"""
A refined implementation of TamGenRL for 3-criteria optimization (QED, SAS, and Binding Affinity),
enhanced with TamGen paper's diversity strategies.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from rdkit import Chem
import torch
from tqdm import tqdm
import random
import regex
import re

from TamGen_Demo import TamGenDemo
from fairseq import progress_bar, utils

from feedback.centroid_optimizer import centroid_shift_optimize
from utils import InjectionMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("tamgen_rl.log"),
        logging.StreamHandler()
    ]
)


class TamGenRL(TamGenDemo):
    """
    Implements a reinforcement learning loop for TamGen, enhanced with multi-configuration
    generation strategies from the original paper for improved diversity.

    Key Features:
    - Multiple pocket definitions with different distance thresholds
    - Varied beta parameters for VAE sampling
    - Both conditional and unconditional generation
    - Scaffold augmentation for seeded generation
    - Multi-seed generation for diversity
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_protein_inputs = {}  # Store multiple configs
        self.device = next(self.models[0].parameters()).device
        self.latent_dim = self._detect_latent_dim()
        self.injection_monitor = InjectionMonitor(self)
        
        # Configuration parameters from the paper
        self.pocket_thresholds = [8, 10, 12, 15]  # Multiple pocket sizes
        self.beta_values = [0.1, 0.5, 1.0]  # VAE beta parameters
        self.use_conditional_modes = [True, False]  # Both VAE and non-VAE
        self.augmentation_rounds = 20  # For scaffold augmentation
        
        # Now log initialization after attributes are defined
        self._log_initialization()
    
    def _detect_latent_dim(self) -> int:
        """Infers the model's latent dimension from its architecture."""
        try:
            if hasattr(self.models[0], 'encoder') and hasattr(self.models[0].encoder, 'vae_encoder'):
                vae_encoder = self.models[0].encoder.vae_encoder
                if hasattr(vae_encoder, 'out_proj'):
                    return vae_encoder.out_proj.out_features // 2
            return 256
        except Exception as e:
            logging.warning(f"Could not automatically detect latent dimension, defaulting to 256: {e}")
            return 256

    def _log_initialization(self):
        """Logs initial configuration details."""
        logging.info(f"TamGenRL initialized on device: {self.device}")
        logging.info(f"Detected latent dimension: {self.latent_dim}")
        logging.info(f"Pocket thresholds: {self.pocket_thresholds}")
        logging.info(f"Beta values: {self.beta_values}")

    def augment_scaffold(self, smiles: str, num_augmentations: int = 20) -> Set[str]:
        """
        Augments a SMILES string by creating multiple equivalent representations.
        This is directly from the paper's approach to increase diversity.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {smiles}
        
        smi_can = Chem.MolToSmiles(mol)
        augmented_smiles = {smi_can, smiles}
        
        remapping = list(range(mol.GetNumAtoms()))
        for i in range(num_augmentations):
            random.shuffle(remapping)
            new_mol = Chem.RenumberAtoms(mol, remapping)
            new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=False)
            m = Chem.MolFromSmiles(new_smiles)
            if m is not None:
                x2 = Chem.MolToSmiles(m)
                if smi_can == x2:
                    augmented_smiles.add(new_smiles)
        
        return augmented_smiles

    def prepare_multi_config_data(self, pdb_id: str, scaffold_smiles: Optional[List[str]] = None):
        """
        Prepares multiple data configurations with different pocket thresholds.
        This mimics the paper's approach of using multiple pocket definitions.
        """
        logging.info(f"Preparing multi-configuration data for {pdb_id}")
        
        configurations = []
        for thr in self.pocket_thresholds:
            subset_name = f"gen_{pdb_id.lower()}_t{thr}"
            try:
                # Try to load the data subset
                self.reload_data(subset=subset_name)
                configurations.append({
                    'threshold': thr,
                    'subset': subset_name,
                    'scaffolds': scaffold_smiles
                })
                logging.info(f"   ✓ Loaded configuration with threshold {thr}Å")
            except Exception as e:
                logging.warning(f"   ✗ Could not load threshold {thr}Å: {e}")
        
        if not configurations:
            # Fallback to single configuration
            logging.warning("No multi-threshold data available, using single configuration")
            subset_name = f"gen_{pdb_id.lower()}"
            self.reload_data(subset=subset_name)
            configurations.append({
                'threshold': 10,
                'subset': subset_name,
                'scaffolds': scaffold_smiles
            })
        
        return configurations

    def _generate_with_config(self, config: Dict, beta: float, use_conditional: bool, 
                             num_samples: int, maxseed: int, use_cuda: bool) -> Tuple[List[str], List[np.ndarray]]:
        """
        Generates molecules with a specific configuration (pocket size, beta, conditional mode).
        """
        # Override model settings for this configuration
        overrides = {
            'sample_beta': beta,
            'gen_vae': use_conditional
        }
        
        # Temporarily update model configuration
        for model in self.models:
            for key, value in overrides.items():
                if hasattr(model, key):
                    setattr(model, key, value)
        
        self.args.sample_beta = beta
        self.args.gen_vae = use_conditional
        
        # Reload data for this configuration
        self.reload_data(subset=config['subset'])
        
        smiles_list = []
        z_vectors_list = []
        
        # Multi-seed generation like in the paper
        seed_list = list(range(1, min(50, maxseed))) + [3407]  # 3407 is a "good" seed from papers
        
        for seed in tqdm(seed_list, desc=f"Config: thr={config['threshold']}, β={beta}, VAE={use_conditional}"):
            if len(smiles_list) >= num_samples:
                break
                
            self._set_seed(seed, use_cuda)
            
            try:
                with progress_bar.build_progress_bar(self.args, self.itr) as t:
                    for sample in t:
                        if len(smiles_list) >= num_samples:
                            break
                            
                        sample = utils.move_to_cuda(sample) if use_cuda else sample
                        if 'net_input' not in sample:
                            continue
                        
                        # Generate and extract
                        batch_results = self._generate_and_extract_latents(sample, use_cuda)
                        
                        for smiles, z in batch_results:
                            if len(smiles_list) < num_samples:
                                smiles_list.append(smiles)
                                z_vectors_list.append(z)
                        
            except Exception as e:
                logging.warning(f"Error with seed {seed}: {e}")
                continue
        
        return smiles_list, z_vectors_list

    def _generate_initial_population_multi_config(self, num_samples: int, maxseed: int, 
                                                 use_cuda: bool, pdb_id: str,
                                                 scaffold_smiles: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generates initial population using multiple configurations for maximum diversity.
        """
        logging.info("🌱 Generating diverse initial population with multiple configurations...")
        
        # Prepare configurations
        configurations = self.prepare_multi_config_data(pdb_id, scaffold_smiles)
        
        # Augment scaffolds if provided
        augmented_scaffolds = set()
        if scaffold_smiles:
            for scaffold in scaffold_smiles:
                augmented = self.augment_scaffold(scaffold, self.augmentation_rounds)
                augmented_scaffolds.update(augmented)
            logging.info(f"   📐 Augmented {len(scaffold_smiles)} scaffolds to {len(augmented_scaffolds)} variants")
        
        all_smiles = []
        all_z_vectors = []
        
        # Calculate samples per configuration
        total_configs = len(configurations) * len(self.beta_values) * len(self.use_conditional_modes)
        samples_per_config = max(10, num_samples // total_configs)
        
        # Generate with all combinations
        config_count = 0
        for config in configurations:
            for beta in self.beta_values:
                for use_conditional in self.use_conditional_modes:
                    config_count += 1
                    logging.info(f"\n🔧 Configuration {config_count}/{total_configs}:")
                    logging.info(f"   Threshold: {config['threshold']}Å, Beta: {beta}, Conditional: {use_conditional}")
                    
                    smiles_batch, z_batch = self._generate_with_config(
                        config, beta, use_conditional, 
                        samples_per_config, maxseed, use_cuda
                    )
                    
                    all_smiles.extend(smiles_batch)
                    all_z_vectors.extend(z_batch)
                    
                    logging.info(f"   Generated {len(smiles_batch)} molecules")
        
        # Convert to arrays
        if all_z_vectors:
            z_vectors = np.stack(all_z_vectors[:num_samples])
            smiles_list = all_smiles[:num_samples]
        else:
            raise RuntimeError("Failed to generate any molecules with multi-configuration approach")
        
        # Report diversity
        unique_count = len(set(smiles_list))
        diversity_ratio = unique_count / len(smiles_list)
        logging.info(f"\n✅ Multi-config generation complete:")
        logging.info(f"   Total molecules: {len(smiles_list)}")
        logging.info(f"   Unique molecules: {unique_count} ({diversity_ratio:.2%})")
        
        return z_vectors, smiles_list

    def sample(self,
               num_samples: int = 1000,  # Increased default
               num_iter: int = 5,
               alpha: float = 0.5,
               top_k: int = 100,  # Increased default
               maxseed: int = 50,  # Increased default
               use_cuda: bool = True,
               batch_size: int = 8,  # Increased default
               save_intermediates: bool = True,
               pdb_id: Optional[str] = None,
               use_binding_affinity: bool = True,
               weights: Optional[Dict[str, float]] = None,
               affinity_config: Optional[Dict[str, float]] = None,
               scaffold_smiles: Optional[List[str]] = None,
               use_multi_config: bool = True,  # New parameter
               **kwargs) -> List[str]:
        """
        Generates and optimizes molecules with enhanced diversity strategies.
        
        Args:
            use_multi_config: If True, uses multiple pocket/beta/conditional configurations
            scaffold_smiles: Optional list of scaffold SMILES for seeded generation
        """
        os.makedirs("latent_logs", exist_ok=True)
        
        weights = weights or {'qed': 1.0, 'sas': 1.0, 'binding_affinity': 3.0}
        
        self._log_optimization_start(num_samples, num_iter, batch_size, weights, pdb_id, use_binding_affinity)
        
        z_vectors, smiles_list, iteration_results = None, None, []
        
        for iteration in range(num_iter):
            iter_start_time = time.time()
            current_alpha = alpha * (0.8 ** iteration)
            
            logging.info(f"\n🔄 Iteration {iteration + 1}/{num_iter} (α={current_alpha:.3f})")
            
            try:
                if iteration == 0:
                    if use_multi_config and pdb_id:
                        # Use multi-configuration generation
                        z_vectors, smiles_list = self._generate_initial_population_multi_config(
                            num_samples, maxseed, use_cuda, pdb_id, scaffold_smiles
                        )
                    else:
                        # Fallback to original single-config generation
                        z_vectors, smiles_list = self._generate_initial_population(
                            num_samples, maxseed, use_cuda
                        )
                else:
                    # Subsequent iterations still use latent space optimization
                    if z_vectors is not None:
                        # For diversity, add some random sampling alongside latent decoding
                        latent_smiles = self._decode_latents_to_smiles(z_vectors[:len(z_vectors)//2], batch_size, use_cuda)
                        
                        # Add fresh random samples for diversity
                        fresh_z, fresh_smiles = self._generate_initial_population(
                            len(z_vectors)//2, maxseed//2, use_cuda
                        )
                        
                        smiles_list = latent_smiles + fresh_smiles
                        z_vectors = np.vstack([z_vectors[:len(latent_smiles)], fresh_z])
                    else:
                        raise RuntimeError("No latent vectors available for generation")
                
                if not smiles_list:
                    raise RuntimeError(f"No valid molecules generated in iteration {iteration + 1}")
                
                # Remove duplicates
                seen_smiles = set()
                unique_smiles_list = []
                unique_indices = []

                for i, smiles in enumerate(smiles_list):
                    if smiles not in seen_smiles:
                        seen_smiles.add(smiles)
                        unique_smiles_list.append(smiles)
                        unique_indices.append(i)
                
                duplicates_removed = len(smiles_list) - len(unique_smiles_list)
                smiles_list = unique_smiles_list
                z_vectors = z_vectors[unique_indices]
                
                if duplicates_removed > 0:
                    logging.info(f"   🗑️  Removed {duplicates_removed} duplicates before reward calculation")
                
                # Continue with optimization
                z_vectors, rewards, metrics = self._optimize_latent_space(
                    z_vectors, smiles_list, current_alpha, min(top_k, len(smiles_list)),
                    iteration, pdb_id, use_binding_affinity, weights, affinity_config
                )

                self.injection_monitor.monitor_injection_quality(iteration, z_vectors, smiles_list, rewards)

                iter_time = time.time() - iter_start_time
                iteration_result = self._record_iteration_metrics(
                    iteration, smiles_list, rewards, metrics, current_alpha, iter_time, pdb_id, use_binding_affinity
                )
                iteration_results.append(iteration_result)
                
                binding_affinities = [m.get('binding_affinity') for m in metrics if m.get('binding_affinity') is not None]
                self._log_iteration_progress(iteration, smiles_list, rewards, binding_affinities, metrics, current_alpha, iter_time)
                
                if save_intermediates:
                    self._save_intermediate_results(iteration + 1, smiles_list, rewards, metrics, z_vectors)
                
                self._cleanup_memory()
                
            except Exception as e:
                logging.error(f"Error in iteration {iteration + 1}: {e}")
                if iteration == 0:
                    raise RuntimeError(f"Failed in initial generation: {e}")
                else:
                    logging.warning("Continuing with previous iteration's results.")
                    break
        
        self._log_final_summary(iteration_results)
        logging.info("🎉 TamGenRL optimization complete!")
        return smiles_list if smiles_list else []

    # Keep all other methods from the original implementation unchanged
    def _log_optimization_start(self, num_samples: int, num_iter: int, batch_size: int, 
                                weights: Dict[str, float], pdb_id: Optional[str], use_binding_affinity: bool):
        """Logs the configuration and parameters at the start of the optimization process."""
        logging.info("🚀 Starting TamGenRL optimization with enhanced diversity")
        logging.info(f"   Target: {num_samples} molecules × {num_iter} iterations")
        logging.info(f"   Device: {self.device}, Batch size: {batch_size}")
        logging.info(f"   Weights: QED={weights['qed']}, SAS={weights['sas']}, Binding={weights['binding_affinity']}")
        logging.info(f"   Multi-config diversity enhancement: ENABLED")
        logging.info(f"   Configurations: {len(self.pocket_thresholds)} thresholds × {len(self.beta_values)} betas × 2 modes")
        
        if use_binding_affinity and pdb_id:
            logging.info(f"   🧬 Binding affinity optimization enabled for PDB: {pdb_id}")
        elif use_binding_affinity:
            logging.info("   📋 Using pre-computed docking scores for binding affinity")
        else:
            logging.info("   ⚗️  Using QED and SAS only")

    def _log_iteration_progress(self, iteration: int, smiles_list: List[str], rewards: List[float], 
                                binding_affinities: List[Optional[float]], metrics: List[Dict], 
                                current_alpha: float, iter_time: float):
        """Logs a summary of progress for the current iteration."""
        unique_count = len(set(smiles_list))
        diversity_ratio = unique_count / len(smiles_list)
        
        logging.info(f"   ✓ Generated {len(smiles_list)} molecules")
        logging.info(f"   ✓ Diversity: {unique_count}/{len(smiles_list)} ({diversity_ratio:.2%})")
        logging.info(f"   ✓ Reward: μ={np.mean(rewards):.3f}, σ={np.std(rewards):.3f}, max={np.max(rewards):.3f}")
        
        if binding_affinities:
            binding_values = [float(x) for x in binding_affinities if x is not None]
            if binding_values:
                success_rate = len(binding_values) / len(metrics)
                best_affinity = min(binding_values)
                logging.info(f"   🧬 Binding affinity: {len(binding_values)}/{len(metrics)} successful ({success_rate:.1%})")
                logging.info(f"   🏆 Best binding: {best_affinity:.2f} kcal/mol")
        
        logging.info(f"   ✓ Time: {iter_time:.1f}s")

    def _record_iteration_metrics(self, iteration: int, smiles_list: List[str], rewards: List[float],
                                  metrics: List[Dict], current_alpha: float, iter_time: float,
                                  pdb_id: Optional[str], use_binding_affinity: bool) -> Dict[str, Any]:
        """Creates a dictionary summarizing the results of a single optimization iteration."""
        result = {
            'iteration': iteration + 1,
            'n_molecules': len(smiles_list),
            'unique_molecules': len(set(smiles_list)),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'alpha': current_alpha,
            'time_seconds': iter_time,
            'pdb_id': pdb_id,
            'use_binding_affinity': use_binding_affinity
        }
        
        binding_affinities = [m.get('binding_affinity') for m in metrics if m.get('binding_affinity') is not None]
        if binding_affinities:
            binding_values = [float(x) for x in binding_affinities if x is not None]
            result.update({
                'binding_affinity_success_rate': len(binding_values) / len(metrics),
                'best_binding_affinity': min(binding_values),
                'mean_binding_affinity': np.mean(binding_values)
            })
        
        return result

    def _generate_initial_population(self, num_samples: int, maxseed: int, use_cuda: bool) -> Tuple[np.ndarray, List[str]]:
        """Generates the initial set of molecules and their latent vectors using the base TamGen model."""
        logging.info("🌱 Generating initial molecular population...")
        smiles_and_latents = []
        
        with tqdm(total=min(num_samples, maxseed * 50), desc="Initial generation") as pbar:
            for seed in range(1, maxseed + 1):
                if len(smiles_and_latents) >= num_samples:
                    break
                
                self._set_seed(seed, use_cuda)
                
                try:
                    with progress_bar.build_progress_bar(self.args, self.itr) as t:
                        for sample in t:
                            if len(smiles_and_latents) >= num_samples:
                                break
                            
                            sample = utils.move_to_cuda(sample) if use_cuda else sample
                            if 'net_input' not in sample:
                                continue
                            
                            if not self.stored_protein_inputs:
                                if isinstance(sample, dict):
                                    self._store_protein_input(sample)
                            
                            if isinstance(sample, dict):
                                batch_results = self._generate_and_extract_latents(sample, use_cuda)
                                smiles_and_latents.extend(batch_results)
                                pbar.update(len(batch_results))
                            
                            if len(smiles_and_latents) >= num_samples:
                                break
                    
                except Exception as e:
                    logging.warning(f"Error during initial generation with seed {seed}: {e}")
                    continue
        
        if not smiles_and_latents:
            raise RuntimeError("Failed to generate any valid molecules in the initial step.")
        if not self.stored_protein_inputs:
            raise RuntimeError("Protein input was not stored during initial generation.")
        
        smiles_list, z_vectors = zip(*smiles_and_latents)
        z_vectors = np.stack(z_vectors)
        
        unique_count = len(set(smiles_list))
        diversity_ratio = unique_count / len(smiles_list)
        logging.info(f"   ✓ Generated {len(smiles_list)} initial molecules")
        logging.info(f"   ✓ Diversity: {unique_count}/{len(smiles_list)} ({diversity_ratio:.2%})")
        
        return z_vectors, list(smiles_list)

    def _set_seed(self, seed: int, use_cuda: bool):
        """Sets random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _store_protein_input(self, sample: Dict[str, Any]):
        """Stores the protein context from a sample for reuse in subsequent generations."""
        config_key = f"default"  # Could be extended to store per-config
        protein_input = {
            'src_tokens': sample['net_input']['src_tokens'].clone(),
            'src_lengths': sample['net_input']['src_lengths'].clone(),
            'src_coord': sample['net_input'].get('src_coord', None).clone() 
                        if sample['net_input'].get('src_coord') is not None else None,
        }
        self.stored_protein_inputs[config_key] = protein_input
        logging.info(f"📦 Stored protein context for generation: {protein_input['src_tokens'].shape}")

    def _generate_and_extract_latents(self, sample: Dict[str, Any], use_cuda: bool) -> List[Tuple[str, np.ndarray]]:
        """Generates molecules from a sample and extracts their corresponding latent representations."""
        results = []
        
        try:
            hypos = self.task.inference_step(self.generator, self.models, sample, None)
            
            model = self.models[0]
            if hasattr(model, 'encoder'):
                encoder_out = model.encoder.forward(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'],
                    src_coord=sample['net_input'].get('src_coord', None),
                    tgt_tokens=sample.get('target', None),
                    tgt_coord=sample['net_input'].get('tgt_coord', None),
                )
                
                if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                    z = encoder_out['latent_mean'].detach().cpu().numpy()
                    
                    if z.ndim == 3:
                        z = z.mean(axis=0)
                    
                    for i, sample_id in enumerate(sample['id'].tolist()):
                        if i >= len(hypos) or len(hypos[i]) == 0 or i >= len(z):
                            continue
                        
                        best_hypo = hypos[i][0]
                        hypo_tokens = best_hypo["tokens"].int().cpu()
                        smiles = self.tgt_dict.string(hypo_tokens, None).strip().replace(" ", "")
                        
                        if Chem.MolFromSmiles(smiles) is not None:
                            results.append((smiles, z[i]))
        
        except Exception as e:
            logging.warning(f"Skipping batch due to error in generation/extraction: {e}")
        
        return results

    def _decode_latents_to_smiles(self, z_vectors: np.ndarray, batch_size: int, use_cuda: bool) -> List[str]:
        """Generates SMILES strings from a batch of latent vectors using the stored protein context."""
        if not self.stored_protein_inputs:
            raise RuntimeError("Cannot generate from latents without a stored protein context.")
        
        logging.info(f"🔄 Generating from {len(z_vectors)} latent vectors...")
        
        # Use the first stored protein input
        protein_input = list(self.stored_protein_inputs.values())[0]
        all_results = []
        
        for start_idx in range(0, len(z_vectors), batch_size):
            end_idx = min(start_idx + batch_size, len(z_vectors))
            batch_z = z_vectors[start_idx:end_idx]
            
            try:
                batch_results = self._generate_batch_with_latent_injection(batch_z, protein_input, use_cuda)
                all_results.extend(batch_results)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    all_results.extend(self._handle_oom_generation(batch_z, protein_input, use_cuda, batch_size))
                else:
                    raise e
            
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        valid_results = [s for s in all_results if s and Chem.MolFromSmiles(s) is not None]
        logging.info(f"   ✓ Generated {len(valid_results)}/{len(z_vectors)} valid SMILES")
        
        return valid_results

    def _handle_oom_generation(self, batch_z: np.ndarray, protein_input: Dict[str, torch.Tensor], 
                              use_cuda: bool, original_batch_size: int) -> List[str]:
        """Handles an out-of-memory error by retrying generation with a smaller batch size."""
        logging.warning(f"OOM detected, reducing batch size from {original_batch_size}")
        results = []
        smaller_batch_size = max(1, original_batch_size // 2)
        
        for sub_start in range(0, len(batch_z), smaller_batch_size):
            sub_end = min(sub_start + smaller_batch_size, len(batch_z))
            sub_batch_z = batch_z[sub_start:sub_end]
            try:
                sub_results = self._generate_batch_with_latent_injection(sub_batch_z, protein_input, use_cuda)
                results.extend(sub_results)
            except Exception:
                logging.error(f"Failed to process sub-batch of size {len(sub_batch_z)}")
                results.extend([""] * (sub_end - sub_start))
        
        return results

    def _generate_batch_with_latent_injection(self, z_batch: np.ndarray, 
                                              protein_input: Dict[str, torch.Tensor], 
                                              use_cuda: bool) -> List[str]:
        """Generates a batch of molecules by injecting latent vectors into the encoder output."""
        # Use random seed for diversity
        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        model = self.models[0]
        model.eval()

        # Keep some dropout for diversity
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.1  # Small dropout for diversity
        
        batch_size = len(z_batch)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=self.device)  
        
        sample = self._prepare_generation_sample(protein_input, batch_size)
        
        try:
            encoder_out = model.encoder.forward(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                src_coord=sample["net_input"]["src_coord"],
                tgt_tokens=None,
                tgt_coord=None
            )
            
            # More sophisticated latent injection with noise
            noise = torch.randn_like(z_tensor) * 0.1  # Add noise for diversity
            z_noisy = z_tensor + noise
            z_expanded = z_noisy.unsqueeze(0).expand(encoder_out['encoder_out'].size(0), -1, -1)
            encoder_out['encoder_out'] = encoder_out['encoder_out'] + z_expanded
            
            sample["encoder_outs_override"] = [encoder_out]
            hypos = self.task.inference_step(self.generator, self.models, sample, None)
            
        except Exception as e:
            raise RuntimeError(f"Latent injection and generation failed: {e}")
        
        results = []
        for i, hypos_i in enumerate(hypos):
            if not hypos_i:
                results.append("")  # Don't fail, just skip
                continue
            
            best_hypo = hypos_i[0]
            hypo_tokens = best_hypo["tokens"].int().cpu()
            smiles = self.tgt_dict.string(hypo_tokens, None).strip().replace(" ", "")
            
            # Don't raise error on invalid SMILES, just skip
            if Chem.MolFromSmiles(smiles) is None:
                results.append("")
            else:
                results.append(smiles)
        
        return results

    def _prepare_generation_sample(self, protein_input: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, Any]:
        """Prepares a sample dictionary for inference by expanding the protein input to match the batch size."""
        src_tokens = protein_input['src_tokens'][:1].expand(batch_size, -1).to(self.device)
        src_lengths = protein_input['src_lengths'][:1].expand(batch_size).to(self.device)
        src_coord = None
        if protein_input.get('src_coord') is not None:
            src_coord = protein_input['src_coord'][:1].expand(batch_size, -1, -1).to(self.device)
        
        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_coord": src_coord,
            },
            "id": torch.arange(batch_size, device=self.device),
        }

    def _optimize_latent_space(self, z_vectors: np.ndarray, smiles_list: List[str], shift_alpha: float,
                               top_k: int, iteration: int, pdb_id: Optional[str] = None,
                               use_binding_affinity: bool = True, 
                               weights: Optional[Dict[str, float]] = None,
                               affinity_config: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
        """Optimizes latent vectors by calculating rewards and shifting them towards a high-reward centroid."""
        logging.info("📊 Optimizing latent space with 3 criteria...")
        
        docking_scores: List[Optional[float]] = [None] * len(smiles_list)
        
        z_shifted, rewards, metrics = centroid_shift_optimize(
            z_vectors=z_vectors,
            smiles_list=smiles_list,
            docking_scores=docking_scores,
            latent_dim=self.latent_dim,
            top_k=min(top_k, len(smiles_list)),
            shift_alpha=shift_alpha,
            noise_sigma=0.05 + 0.01 * iteration,
            device="auto",
            epochs=50,
            pdb_id=pdb_id,
            use_binding_affinity=use_binding_affinity,
            weights=weights,
            affinity_config=affinity_config
        )
        
        return z_shifted, rewards, metrics

    def _save_intermediate_results(self, iteration: int, smiles_list: List[str], rewards: List[float],
                                   metrics: List[Dict[str, Any]], z_vectors: np.ndarray):
        """Saves the SMILES, rewards, and latent vectors from an iteration to disk."""
        with open(f"latent_logs/results_iter_{iteration}.tsv", "w") as f:
            f.write("SMILES\tReward\tQED\tSAS\tBinding_Affinity\n")
            for smi, reward, metric in zip(smiles_list, rewards, metrics):
                qed = metric.get('qed', '')
                sas = metric.get('sas', '')
                binding_affinity = metric.get('binding_affinity', '')
                f.write(f"{smi}\t{reward:.4f}\t{qed}\t{sas}\t{binding_affinity}\n")
        
        np.savetxt(f"latent_logs/latents_iter_{iteration}.tsv", z_vectors, fmt="%.6f")
        logging.info(f"   ✓ Saved results for iteration {iteration}")

    def _cleanup_memory(self):
        """Releases unused CUDA memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _log_final_summary(self, iteration_results: List[Dict[str, Any]]):
        """Logs a summary table of the entire optimization process."""
        if not iteration_results:
            return
        
        logging.info("\n📈 Final Optimization Summary:")
        logging.info("   Iteration | Molecules | Unique | Mean Reward | Max Reward | Best Binding | Time")
        logging.info("   ---------|-----------|--------|-------------|------------|--------------|-----")
        
        for result in iteration_results:
            best_binding = result.get('best_binding_affinity', '')
            best_binding_str = f"{best_binding:10.2f}" if best_binding != '' else "        N/A"
                
            logging.info(f"   {result['iteration']:8d} | "
                        f"{result['n_molecules']:9d} | "
                        f"{result['unique_molecules']:6d} | "
                        f"{result['mean_reward']:11.3f} | "
                        f"{result['max_reward']:10.3f} | "
                        f"{best_binding_str} | "
                        f"{result['time_seconds']:4.1f}s")
        
        self._log_overall_stats(iteration_results)

    def _log_overall_stats(self, iteration_results: List[Dict[str, Any]]):
        """Logs high-level statistics about the overall optimization performance."""
        final_result = iteration_results[-1]
        initial_result = iteration_results[0]
        
        reward_improvement = final_result['mean_reward'] - initial_result['mean_reward']
        diversity_final = final_result['unique_molecules'] / final_result['n_molecules']
        total_time = sum(r['time_seconds'] for r in iteration_results)
        
        logging.info(f"\n   💡 Reward improvement: {reward_improvement:+.3f}")
        logging.info(f"   🎯 Final diversity: {diversity_final:.2%}")
        logging.info(f"   ⏱️  Total time: {total_time:.1f}s")
        
        if final_result.get('best_binding_affinity'):
            logging.info(f"   🧬 Best binding affinity: {final_result['best_binding_affinity']:.2f} kcal/mol")
            if final_result.get('binding_affinity_success_rate'):
                logging.info(f"   📊 Final docking success rate: {final_result['binding_affinity_success_rate']:.1%}")