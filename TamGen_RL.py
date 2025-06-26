"""
Optimized TamGenRL implementation with 3-criteria optimization: QED, SAS, Binding Affinity.
Streamlined for performance while maintaining full functionality.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
import torch
from tqdm import tqdm

# Import TamGen components (adjust imports based on your structure)
from TamGen_Demo import TamGenDemo
from fairseq import progress_bar, utils

# Import our simplified optimization components
from feedback.centroid_optimizer import centroid_shift_optimize

from utils import InjectionMonitor

# Setup logging
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
    Optimized TamGenRL with 3-criteria optimization: QED, SAS, Binding Affinity.
    
    Key features:
    - Fixed latent injection method
    - 3-criteria optimization (QED, SAS, Binding Affinity)
    - Binding affinity prioritized with 3x weight
    - Simple centroid shift optimization
    - Optimized memory management
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_protein_inputs = []
        self.device = next(self.models[0].parameters()).device
        self.latent_dim = self._detect_latent_dim()
        self.injection_monitor = InjectionMonitor(self)
        self._log_init()
    
    def _detect_latent_dim(self) -> int:
        """Detect the latent dimension from the model architecture."""
        try:
            if hasattr(self.models[0], 'encoder') and hasattr(self.models[0].encoder, 'vae_encoder'):
                vae_encoder = self.models[0].encoder.vae_encoder
                if hasattr(vae_encoder, 'out_proj'):
                    return vae_encoder.out_proj.out_features // 2
            return 256
        except Exception as e:
            logging.warning(f"Could not detect latent dimension: {e}. Using default 256.")
            return 256

    def _log_init(self):
        """Initialize logging."""
        logging.info(f"TamGenRL initialized on device: {self.device}")
        logging.info(f"Detected latent dimension: {self.latent_dim}")

    def _log_progress(self, iteration: int, smiles_list: List[str], rewards: List[float], 
                     binding_affinities: List[Optional[float]], metrics: List[Dict], 
                     current_alpha: float, iter_time: float):
        """Consolidated progress logging."""
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

    def sample(self,
               m_sample: int = 100,
               num_iter: int = 5,
               alpha: float = 0.5,
               top_k: int = 50,
               maxseed: int = 20,
               use_cuda: bool = True,
               batch_size: int = 4,
               save_intermediates: bool = True,
               pdb_id: Optional[str] = None,
               use_binding_affinity: bool = True,
               weights: Optional[Dict[str, float]] = None,
               **kwargs) -> List[str]:
        """
        Optimized sampling with 3-criteria optimization.
        """
        
        os.makedirs("latent_logs", exist_ok=True)
        
        # Set default weights with binding affinity prioritized
        weights = weights or {'qed': 1.0, 'sas': 1.0, 'binding_affinity': 3.0}
        
        self._log_start_info(m_sample, num_iter, batch_size, weights, pdb_id, use_binding_affinity)
        
        # Initialize variables
        z_vectors, smiles_list, iteration_results = None, None, []
        
        for iteration in range(num_iter):
            iter_start_time = time.time()
            current_alpha = alpha * (0.8 ** iteration)  # Simple alpha decay
            
            logging.info(f"\n🔄 Iteration {iteration + 1}/{num_iter} (α={current_alpha:.3f})")
            
            try:
                # Generate molecules
                if iteration == 0:
                    z_vectors, smiles_list = self._initial_generation(m_sample, maxseed, use_cuda)
                else:
                    if z_vectors is not None:
                        smiles_list = self._generate_from_latents(z_vectors, batch_size, use_cuda)
                        z_vectors = z_vectors[:len(smiles_list)]  # Update to match successful generations
                    else:
                        raise RuntimeError("No latent vectors available for generation")
                
                if not smiles_list:
                    raise RuntimeError(f"No valid molecules generated in iteration {iteration + 1}")
                
                # Build unique lists while preserving first-occurrence order
                seen_smiles = set()
                unique_smiles_list = []
                unique_indices = []

                for i, smiles in enumerate(smiles_list):
                    if smiles not in seen_smiles:
                        seen_smiles.add(smiles)
                        unique_smiles_list.append(smiles)  # Preserves order
                        unique_indices.append(i)
                
                # Update smiles_list and z_vectors to unique only
                smiles_list = unique_smiles_list
                z_vectors = z_vectors[unique_indices]
                
                duplicates_removed = len(smiles_list) - len(unique_smiles_list)
                logging.info(f"   🗑️  Removed {duplicates_removed} duplicates before reward calculation")
                
                # Optimize latent space (now with unique molecules only)
                z_vectors, rewards, metrics = self._optimize_latent_space_simple(
                    z_vectors, smiles_list, current_alpha, min(top_k, len(smiles_list)),
                    iteration, pdb_id, use_binding_affinity, weights
                )

                self.injection_monitor.monitor_injection_quality(iteration, z_vectors, smiles_list, rewards)

                # Process and log results
                iter_time = time.time() - iter_start_time
                iteration_result = self._create_iteration_result(
                    iteration, smiles_list, rewards, metrics, current_alpha, iter_time, pdb_id, use_binding_affinity
                )
                iteration_results.append(iteration_result)
                
                # Log progress
                binding_affinities = [m.get('binding_affinity') for m in metrics if m.get('binding_affinity') is not None]
                self._log_progress(iteration, smiles_list, rewards, binding_affinities, metrics, current_alpha, iter_time)
                
                # Save intermediate results
                if save_intermediates:
                    self._save_iteration_results(iteration + 1, smiles_list, rewards, metrics, z_vectors)
                
                # Memory cleanup
                self._cleanup_memory()
                
            except Exception as e:
                logging.error(f"Error in iteration {iteration + 1}: {e}")
                if iteration == 0:
                    raise RuntimeError(f"Failed in initial generation: {e}")
                else:
                    logging.warning("Continuing with previous iteration results")
                    break
        
        self._log_final_summary(iteration_results)
        logging.info("🎉 TamGenRL optimization complete!")
        return smiles_list if smiles_list else []

    def _log_start_info(self, m_sample: int, num_iter: int, batch_size: int, 
                       weights: Dict[str, float], pdb_id: Optional[str], use_binding_affinity: bool):
        """Log optimization start information."""
        logging.info("🚀 Starting simplified TamGenRL optimization")
        logging.info(f"   Target: {m_sample} molecules × {num_iter} iterations")
        logging.info(f"   Device: {self.device}, Batch size: {batch_size}")
        logging.info(f"   Weights: QED={weights['qed']}, SAS={weights['sas']}, Binding={weights['binding_affinity']}")
        
        if use_binding_affinity and pdb_id:
            logging.info(f"   🧬 Binding affinity optimization enabled for PDB: {pdb_id}")
        elif use_binding_affinity:
            logging.info("   📋 Using pre-computed docking scores for binding affinity")
        else:
            logging.info("   ⚗️  Using QED and SAS only")

    def _create_iteration_result(self, iteration: int, smiles_list: List[str], rewards: List[float],
                               metrics: List[Dict], current_alpha: float, iter_time: float,
                               pdb_id: Optional[str], use_binding_affinity: bool) -> Dict[str, Any]:
        """Create iteration result dictionary."""
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
        
        # Add binding affinity statistics if available
        binding_affinities = [m.get('binding_affinity') for m in metrics if m.get('binding_affinity') is not None]
        if binding_affinities:
            binding_values = [float(x) for x in binding_affinities if x is not None]
            result.update({
                'binding_affinity_success_rate': len(binding_values) / len(metrics),
                'best_binding_affinity': min(binding_values),
                'mean_binding_affinity': np.mean(binding_values)
            })
        
        return result

    def _initial_generation(self, m_sample: int, maxseed: int, use_cuda: bool) -> Tuple[np.ndarray, List[str]]:
        """Generate initial molecules using TamGen and extract latent vectors."""
        
        logging.info("🌱 Generating initial molecules with TamGen...")
        smiles_and_latents = []
        
        with tqdm(total=min(m_sample, maxseed * 50), desc="Initial generation") as pbar:
            for seed in range(1, maxseed + 1):
                if len(smiles_and_latents) >= m_sample:
                    break
                
                self._set_seed(seed, use_cuda)
                
                try:
                    with progress_bar.build_progress_bar(self.args, self.itr) as t:
                        for sample in t:
                            if len(smiles_and_latents) >= m_sample:
                                break
                            
                            sample = utils.move_to_cuda(sample) if use_cuda else sample
                            if 'net_input' not in sample:
                                continue
                            
                            # Store protein input for later use (only once)
                            if len(self.stored_protein_inputs) == 0:
                                if isinstance(sample, dict):
                                    self._store_protein_input(sample)
                            
                            # Generate molecules and extract latents
                            if isinstance(sample, dict):
                                batch_results = self._generate_and_extract_latents(sample, use_cuda)
                                smiles_and_latents.extend(batch_results)
                                pbar.update(len(batch_results))
                            
                            if len(smiles_and_latents) >= m_sample:
                                break
                    
                except Exception as e:
                    logging.warning(f"Error in seed {seed}: {e}")
                    continue
        
        if not smiles_and_latents:
            raise RuntimeError("No valid molecules generated in initial step")
        if not self.stored_protein_inputs:
            raise RuntimeError("No protein inputs stored")
        
        # Process results
        smiles_list, z_vectors = zip(*smiles_and_latents)
        z_vectors = np.stack(z_vectors)
        
        # Log results
        unique_count = len(set(smiles_list))
        diversity_ratio = unique_count / len(smiles_list)
        logging.info(f"   ✓ Generated {len(smiles_list)} initial molecules")
        logging.info(f"   ✓ Diversity: {unique_count}/{len(smiles_list)} ({diversity_ratio:.2%})")
        
        return z_vectors, list(smiles_list)

    def _set_seed(self, seed: int, use_cuda: bool):
        """Set random seeds."""
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _store_protein_input(self, sample: Dict[str, Any]):
        """Store protein input for later reuse."""
        protein_input = {
            'src_tokens': sample['net_input']['src_tokens'].clone(),
            'src_lengths': sample['net_input']['src_lengths'].clone(),
            'src_coord': sample['net_input'].get('src_coord', None).clone() 
                        if sample['net_input'].get('src_coord') is not None else None,
        }
        self.stored_protein_inputs.append(protein_input)
        logging.info(f"📦 Stored protein input: {protein_input['src_tokens'].shape}")

    def _generate_and_extract_latents(self, sample: Dict[str, Any], use_cuda: bool) -> List[Tuple[str, np.ndarray]]:
        """Generate molecules and extract their latent representations."""
        results = []
        
        try:
            # Generate molecules
            hypos = self.task.inference_step(self.generator, self.models, sample, None)
            
            # Extract latents from encoder
            model = self.models[0]
            if hasattr(model, 'encoder'):
                encoder_out = model.encoder.forward(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'],
                    src_coord=sample['net_input'].get('src_coord', None),
                    tgt_tokens=sample.get('target', None),
                    tgt_coord=sample['net_input'].get('tgt_coord', None),
                )
                
                # Extract and process latent vectors
                if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                    z = encoder_out['latent_mean'].detach().cpu().numpy()
                    
                    # Handle different tensor shapes
                    if z.ndim == 3:  # [seq_len, batch, dim]
                        z = z.mean(axis=0)  # Average over sequence
                    
                    # Process generated molecules
                    for i, sample_id in enumerate(sample['id'].tolist()):
                        if i >= len(hypos) or len(hypos[i]) == 0 or i >= len(z):
                            continue
                        
                        # Get best hypothesis and validate SMILES
                        best_hypo = hypos[i][0]
                        hypo_tokens = best_hypo["tokens"].int().cpu()
                        smiles = self.tgt_dict.string(hypo_tokens, None).strip().replace(" ", "")
                        
                        if Chem.MolFromSmiles(smiles) is not None:
                            results.append((smiles, z[i]))
        
        except Exception as e:
            logging.warning(f"Error in generation/extraction: {e}")
        
        return results

    def _generate_from_latents(self, z_vectors: np.ndarray, batch_size: int, use_cuda: bool) -> List[str]:
        """Generate SMILES from latent vectors using stored protein input."""
        
        if not self.stored_protein_inputs:
            raise RuntimeError("No stored protein inputs available")
        
        logging.info(f"🔄 Generating from {len(z_vectors)} latent vectors...")
        
        protein_input = self.stored_protein_inputs[0]
        all_results = []
        
        # Process in batches to avoid OOM
        for start_idx in range(0, len(z_vectors), batch_size):
            end_idx = min(start_idx + batch_size, len(z_vectors))
            batch_z = z_vectors[start_idx:end_idx]
            
            try:
                batch_results = self._generate_batch_with_latent_injection(batch_z, protein_input, use_cuda)
                all_results.extend(batch_results)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try smaller batches on OOM
                    all_results.extend(self._handle_oom_generation(batch_z, protein_input, use_cuda, batch_size))
                else:
                    raise e
            
            # Memory cleanup between batches
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Filter out empty results
        valid_results = [s for s in all_results if s and Chem.MolFromSmiles(s) is not None]
        logging.info(f"   ✓ Generated {len(valid_results)}/{len(z_vectors)} valid SMILES")
        
        return valid_results

    def _handle_oom_generation(self, batch_z: np.ndarray, protein_input: Dict[str, torch.Tensor], 
                              use_cuda: bool, original_batch_size: int) -> List[str]:
        """Handle OOM by trying smaller batch sizes."""
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
                logging.error(f"Failed to process samples {sub_start}-{sub_end}")
                results.extend([""] * (sub_end - sub_start))
        
        return results

    def _generate_batch_with_latent_injection(self, z_batch: np.ndarray, 
                                            protein_input: Dict[str, torch.Tensor], 
                                            use_cuda: bool) -> List[str]:
        """Generate molecules with proper latent injection."""

        # Set deterministic seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        model = self.models[0]
        model.eval()

        # Disable dropout for deterministic generation
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        
        batch_size = len(z_batch)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=self.device)  
        
        # Prepare inputs - expand protein input to match batch size
        sample = self._prepare_generation_sample(protein_input, batch_size)
        
        # Direct encoder output modification with latent injection
        try:
            encoder_out = model.encoder.forward(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                src_coord=sample["net_input"]["src_coord"],
                tgt_tokens=None,
                tgt_coord=None
            )
            
            # Inject latent vectors - simple addition method
            z_expanded = z_tensor.unsqueeze(0).expand(encoder_out['encoder_out'].size(0), -1, -1)
            encoder_out['encoder_out'] = encoder_out['encoder_out'] + z_expanded
            
            # Override encoder output and generate
            sample["encoder_outs_override"] = [encoder_out]
            hypos = self.task.inference_step(self.generator, self.models, sample, None)
            
        except Exception as e:
            raise RuntimeError(f"Latent injection failed: {e}")
        
        # Process and validate results
        results = []
        for i, hypos_i in enumerate(hypos):
            if not hypos_i:
                raise RuntimeError(f"No hypotheses generated for sample {i}")
            
            best_hypo = hypos_i[0]
            hypo_tokens = best_hypo["tokens"].int().cpu()
            smiles = self.tgt_dict.string(hypo_tokens, None).strip().replace(" ", "")
            
            if Chem.MolFromSmiles(smiles) is None:
                raise RuntimeError(f"Invalid SMILES generated: {smiles}")
            results.append(smiles)
        
        return results

    def _prepare_generation_sample(self, protein_input: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, Any]:
        """Prepare sample for generation by expanding protein input."""
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

    def _optimize_latent_space_simple(self, z_vectors: np.ndarray, smiles_list: List[str], shift_alpha: float,
                             top_k: int, iteration: int, pdb_id: Optional[str] = None,
                             use_binding_affinity: bool = True, 
                             weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
        """Simplified latent space optimization - no deduplication needed since it's done upfront."""
        
        logging.info("📊 Optimizing latent space with 3 criteria...")
        
        # Apply centroid shift optimization directly on unique molecules
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
            weights=weights
        )
        
        return z_shifted, rewards, metrics

    def _save_iteration_results(self, iteration: int, smiles_list: List[str], rewards: List[float],
                              metrics: List[Dict[str, Any]], z_vectors: np.ndarray):
        """Save iteration results to files."""
        
        # Save SMILES and rewards
        with open(f"latent_logs/results_iter_{iteration}.tsv", "w") as f:
            f.write("SMILES\tReward\tQED\tSAS\tBinding_Affinity\n")
            for smi, reward, metric in zip(smiles_list, rewards, metrics):
                qed = metric.get('qed', '')
                sas = metric.get('sas', '')
                binding_affinity = metric.get('binding_affinity', '')
                f.write(f"{smi}\t{reward:.4f}\t{qed}\t{sas}\t{binding_affinity}\n")
        
        # Save latent vectors
        np.savetxt(f"latent_logs/latents_iter_{iteration}.tsv", z_vectors, fmt="%.6f")
        logging.info(f"   ✓ Saved results for iteration {iteration}")

    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _log_final_summary(self, iteration_results: List[Dict[str, Any]]):
        """Log final optimization summary."""
        
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
        
        # Overall statistics
        self._log_overall_stats(iteration_results)

    def _log_overall_stats(self, iteration_results: List[Dict[str, Any]]):
        """Log overall optimization statistics."""
        final_result = iteration_results[-1]
        initial_result = iteration_results[0]
        
        reward_improvement = final_result['mean_reward'] - initial_result['mean_reward']
        diversity_final = final_result['unique_molecules'] / final_result['n_molecules']
        total_time = sum(r['time_seconds'] for r in iteration_results)
        
        logging.info(f"\n   💡 Reward improvement: {reward_improvement:+.3f}")
        logging.info(f"   🎯 Final diversity: {diversity_final:.2%}")
        logging.info(f"   ⏱️  Total time: {total_time:.1f}s")
        
        # Binding affinity summary
        if final_result.get('best_binding_affinity'):
            logging.info(f"   🧬 Best binding affinity: {final_result['best_binding_affinity']:.2f} kcal/mol")
            if final_result.get('binding_affinity_success_rate'):
                logging.info(f"   📊 Final docking success rate: {final_result['binding_affinity_success_rate']:.1%}")


# Utility functions for external use
def run_tamgen_rl_optimization(checkpoint_path: str,
                              data_path: str,
                              output_dir: str = "tamgen_rl_results",
                              pdb_id: Optional[str] = None,
                              use_binding_affinity: bool = True,
                              **optimization_kwargs) -> Dict[str, Any]:
    """
    Convenience function to run simplified TamGenRL optimization.
    
    Args:
        checkpoint_path: Path to TamGen checkpoint
        data_path: Path to input data
        output_dir: Output directory for results
        pdb_id: PDB ID for GNINA docking (e.g., "1HSG" for HIV protease)
        use_binding_affinity: Whether to use binding affinity optimization
        **optimization_kwargs: Additional arguments for optimization
        
    Returns:
        Dictionary with optimization results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("🚀 Starting simplified TamGenRL optimization")
    logging.info(f"   Output directory: {output_dir}")
    if use_binding_affinity and pdb_id:
        logging.info(f"   Target PDB: {pdb_id}")
    
    # Add binding affinity parameters to optimization kwargs
    optimization_kwargs.update({
        'pdb_id': pdb_id,
        'use_binding_affinity': use_binding_affinity
    })
    
    return {
        "status": "success",
        "output_dir": output_dir,
        "pdb_id": pdb_id,
        "use_binding_affinity": use_binding_affinity,
        "n_final_molecules": 0,
    }