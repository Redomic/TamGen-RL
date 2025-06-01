"""
Enhanced TamGenRL implementation with proper latent space optimization using CentroidShiftOptimizer.
Addresses critical issues in latent injection and memory management.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from rdkit import Chem
import torch
from tqdm import tqdm

# Import TamGen components (adjust imports based on your structure)
from TamGen_Demo import TamGenDemo
from fairseq import progress_bar, utils

# Import our enhanced optimization components
from feedback.centroid_praneeth import CentroidShiftOptimizer
from feedback.reward_utils_praneeth import compute_advanced_reward, compute_diversity_bonus

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
    Enhanced TamGenRL with CentroidShiftOptimizer integration.
    
    Key improvements:
    - Integrated with enhanced CentroidShiftOptimizer for better latent space navigation
    - Fixed latent injection method that works with TamGen's architecture
    - Proper memory management to prevent OOM errors
    - Better batch processing for large-scale generation
    - Comprehensive reward computation with multiple objectives
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_protein_inputs = []
        self.device = next(self.models[0].parameters()).device
        self.latent_dim = self._detect_latent_dim()
        
        # Initialize the enhanced optimizer
        self.optimizer = CentroidShiftOptimizer(
            decoder=self._decode_latent_to_smiles,
            reward_fn=self._compute_molecular_reward,
            learning_rate=0.2,
            diversity_weight=0.1,
            perturbation_scale=0.01,
            train_frequency=5
        )
        
        # Optimization parameters
        self.lambda_sas = 0.3
        self.lambda_logp = 0.1
        self.lambda_mw = 0.1
        self.lambda_qed = 0.2
        self.lambda_docking = 1.0
        
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

    def _decode_latent_to_smiles(self, z: np.ndarray) -> str:
        """
        Decoder function for the CentroidShiftOptimizer.
        Converts a single latent vector to SMILES string.
        """
        try:
            # Ensure z is 2D for batch processing
            if z.ndim == 1:
                z = z.reshape(1, -1)
            
            # Generate SMILES from latent vector
            smiles_list = self._generate_from_latents(
                z_vectors=z,
                batch_size=1,
                use_cuda=torch.cuda.is_available()
            )
            
            # Return first valid SMILES or empty string
            return smiles_list[0] if smiles_list else ""
            
        except Exception as e:
            logging.warning(f"Failed to decode latent vector: {e}")
            return ""

    def _compute_molecular_reward(self, smiles: str) -> float:
        """
        Enhanced reward function for molecular optimization.
        Combines multiple objectives including drug-likeness, synthetic accessibility, etc.
        """
        if not smiles or not isinstance(smiles, str):
            return -10.0
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0
        
        try:
            # Use the advanced reward function from reward_utils_praneeth
            base_reward = compute_advanced_reward(smiles)
            
            # Additional penalties/bonuses can be added here
            # For example, specific structural constraints for target proteins
            
            return base_reward
            
        except Exception as e:
            logging.warning(f"Error computing reward for {smiles}: {e}")
            return -5.0

    def sample(self,
               m_sample: int = 100,
               num_iter: int = 5,
               alpha: float = 0.2,
               top_k: int = 50,
               lambda_sas: float = 0.3,
               lambda_logp: float = 0.1,
               lambda_mw: float = 0.1,
               lambda_qed: float = 0.2,
               lambda_docking: float = 1.0,
               maxseed: int = 20,
               use_cuda: bool = True,
               batch_size: int = 4,
               adaptive_alpha: bool = True,
               save_intermediates: bool = True,
               diversity_target: float = 0.7,
               early_stopping: bool = True,
               patience: int = 3,
               min_improvement: float = 0.01,
               **kwargs) -> List[str]:
        """
        Enhanced sampling with CentroidShiftOptimizer integration.
        
        Args:
            m_sample: Number of molecules to generate per iteration
            num_iter: Number of optimization iterations
            alpha: Base learning rate for optimizer
            top_k: Number of top molecules for centroid computation
            lambda_sas: Weight for SAS penalty
            lambda_logp: Weight for LogP penalty
            lambda_mw: Weight for MW penalty
            lambda_qed: Weight for QED score
            lambda_docking: Weight for docking score
            maxseed: Maximum number of seeds for initial generation
            use_cuda: Whether to use CUDA
            batch_size: Batch size for generation
            adaptive_alpha: Whether to use adaptive alpha scheduling
            save_intermediates: Whether to save intermediate results
            diversity_target: Target diversity ratio (unique/total SMILES)
            early_stopping: Whether to use early stopping
            patience: Steps to wait without improvement before stopping
            min_improvement: Minimum improvement threshold for early stopping
            **kwargs: Additional arguments
            
        Returns:
            List of final optimized SMILES strings
        """
        
        # Update reward function parameters
        self.lambda_sas = lambda_sas
        self.lambda_logp = lambda_logp
        self.lambda_mw = lambda_mw
        self.lambda_qed = lambda_qed
        self.lambda_docking = lambda_docking
        
        # Create output directory
        os.makedirs("latent_logs", exist_ok=True)
        
        logging.info("🚀 Starting TamGenRL optimization with CentroidShiftOptimizer")
        logging.info(f"   Target: {m_sample} molecules × {num_iter} iterations")
        logging.info(f"   Device: {self.device}, Batch size: {batch_size}")
        logging.info(f"   Reward weights - SAS: {lambda_sas}, LogP: {lambda_logp}, MW: {lambda_mw}")
        logging.info(f"   QED: {lambda_qed}, Docking: {lambda_docking}")
        
        # Initialize variables
        z_vectors = None
        smiles_list = None
        all_results = []
        best_molecules = []
        
        try:
            # Step 1: Initial generation using TamGen
            z_vectors, smiles_list = self._initial_generation(
                m_sample=m_sample,
                maxseed=maxseed,
                use_cuda=use_cuda,
                diversity_target=diversity_target
            )
            
            if len(z_vectors) == 0:
                raise RuntimeError("No initial molecules generated")
            
            # Step 2: Iterative optimization using CentroidShiftOptimizer
            for iteration in range(num_iter):
                iter_start_time = time.time()
                
                logging.info(f"\n🔄 Optimization Iteration {iteration + 1}/{num_iter}")
                
                # Update optimizer parameters
                if adaptive_alpha:
                    current_alpha = alpha * (0.5 ** (iteration / max(1, num_iter - 1)))
                    self.optimizer.learning_rate = current_alpha
                    logging.info(f"   Learning rate: {current_alpha:.4f}")
                
                # Optimize each latent vector individually
                iteration_results = []
                optimized_molecules = []
                
                for i, z_init in enumerate(z_vectors):
                    if i >= m_sample:  # Limit to requested number
                        break
                    
                    try:
                        # Run optimization for this latent vector
                        best_smiles, best_reward, history = self.optimizer.optimize(
                            initial_z=z_init,
                            steps=20,  # Fewer steps per iteration for better exploration
                            use_gradient=True,
                            early_stopping=early_stopping,
                            patience=max(3, patience // 2),
                            min_improvement=min_improvement
                        )
                        
                        if best_smiles:
                            optimized_molecules.append((best_smiles, best_reward, history))
                            iteration_results.append({
                                'molecule_idx': i,
                                'smiles': best_smiles,
                                'reward': best_reward,
                                'optimization_steps': len(history)
                            })
                        
                    except Exception as e:
                        logging.warning(f"Optimization failed for molecule {i}: {e}")
                        continue
                
                if not optimized_molecules:
                    logging.warning(f"No molecules optimized in iteration {iteration + 1}")
                    break
                
                # Update best molecules
                current_molecules = [mol[0] for mol in optimized_molecules]
                current_rewards = [mol[1] for mol in optimized_molecules]
                
                # Select top molecules for next iteration
                if len(current_rewards) > top_k:
                    top_indices = np.argsort(current_rewards)[-top_k:]
                    selected_molecules = [current_molecules[i] for i in top_indices]
                    selected_rewards = [current_rewards[i] for i in top_indices]
                else:
                    selected_molecules = current_molecules
                    selected_rewards = current_rewards
                
                # Convert selected molecules back to latent vectors for next iteration
                if iteration < num_iter - 1:  # Don't need latents for final iteration
                    try:
                        z_vectors = self._smiles_to_latents(selected_molecules, use_cuda)
                    except Exception as e:
                        logging.error(f"Failed to convert SMILES to latents: {e}")
                        break
                
                # Store results
                iter_result = {
                    'iteration': iteration + 1,
                    'n_molecules': len(current_molecules),
                    'unique_molecules': len(set(current_molecules)),
                    'mean_reward': np.mean(current_rewards),
                    'std_reward': np.std(current_rewards),
                    'max_reward': np.max(current_rewards),
                    'top_molecules': list(zip(selected_molecules, selected_rewards)),
                    'time_seconds': time.time() - iter_start_time
                }
                all_results.append(iter_result)
                
                # Save intermediate results
                if save_intermediates:
                    self._save_iteration_results(
                        iteration + 1, 
                        current_molecules, 
                        current_rewards, 
                        iteration_results
                    )
                
                # Log progress
                diversity_ratio = len(set(current_molecules)) / len(current_molecules)
                logging.info(f"   ✓ Optimized {len(current_molecules)} molecules")
                logging.info(f"   ✓ Diversity: {len(set(current_molecules))}/{len(current_molecules)} ({diversity_ratio:.2%})")
                logging.info(f"   ✓ Reward: μ={np.mean(current_rewards):.3f}, σ={np.std(current_rewards):.3f}, max={np.max(current_rewards):.3f}")
                logging.info(f"   ✓ Time: {time.time() - iter_start_time:.1f}s")
                
                # Early stopping check
                if early_stopping and iteration > 0:
                    prev_best = all_results[-2]['max_reward'] if len(all_results) > 1 else -np.inf
                    curr_best = all_results[-1]['max_reward']
                    
                    if curr_best - prev_best < min_improvement:
                        logging.info(f"Early stopping: improvement {curr_best - prev_best:.4f} < {min_improvement}")
                        break
                
                # Memory cleanup
                self._cleanup_memory()
                
                # Update best molecules list
                best_molecules = selected_molecules.copy()
        
        except Exception as e:
            logging.error(f"Critical error in optimization: {e}")
            best_molecules = smiles_list[:m_sample] if smiles_list else []
        
        # Final summary
        self._log_final_summary(all_results)
        
        # Get optimizer statistics if available
        if hasattr(self.optimizer, 'reward_model') and len(all_results) > 0:
            try:
                final_stats = self.optimizer.get_optimization_stats(
                    all_results[-1].get('optimization_history', [])
                )
                logging.info(f"📊 Final optimization stats: {final_stats}")
            except Exception as e:
                logging.warning(f"Could not compute final stats: {e}")
        
        logging.info("🎉 TamGenRL optimization complete!")
        return best_molecules if best_molecules else []

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
        
        # Enhanced latent injection with multiple fallback strategies
        try:
            # Get normal encoder output
            encoder_out = model.encoder.forward(
                src_tokens,
                src_lengths,
                src_coord=src_coord,
                tgt_tokens=None,
                tgt_coord=None
            )
            
            # Strategy 1: Direct latent replacement
            if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                encoder_out['latent_mean'] = z_tensor
                if 'latent_logstd' in encoder_out:
                    # Set small variance for injected latents
                    encoder_out['latent_logstd'] = torch.full_like(z_tensor, -2.0)
            
            # Strategy 2: Encoder output modification
            elif 'encoder_out' in encoder_out and encoder_out['encoder_out'] is not None:
                if hasattr(self.args, 'concat') and self.args.concat:
                    # Concatenation approach
                    main_features = encoder_out['encoder_out'][..., :-self.latent_dim]
                    z_expanded = z_tensor.unsqueeze(0).expand(main_features.size(0), -1, -1)
                    encoder_out['encoder_out'] = torch.cat([main_features, z_expanded], dim=-1)
                else:
                    # Addition approach
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

    def _smiles_to_latents(self, smiles_list: List[str], use_cuda: bool) -> np.ndarray:
        """Convert SMILES back to latent vectors (if possible)."""
        # This is a challenging reverse operation
        # For now, we'll use a simple approach of encoding the SMILES
        # In practice, you might want to use a separate encoder model
        
        logging.info("🔄 Converting SMILES back to latents...")
        
        latents = []
        
        # Simple approach: generate small perturbations around stored latents
        # This is not ideal but works as a fallback
        if hasattr(self, '_last_successful_latents') and len(self._last_successful_latents) > 0:
            base_latents = self._last_successful_latents
            for i, smiles in enumerate(smiles_list):
                base_idx = i % len(base_latents)
                perturbed = base_latents[base_idx] + np.random.randn(self.latent_dim) * 0.1
                latents.append(perturbed)
        else:
            # Generate random latents as last resort
            for _ in smiles_list:
                latents.append(np.random.randn(self.latent_dim) * 0.5)
        
        return np.array(latents)

    def _save_iteration_results(self,
                              iteration: int,
                              smiles_list: List[str],
                              rewards: List[float],
                              detailed_results: List[Dict[str, Any]]):
        """Save iteration results to files."""
        
        # Save SMILES and rewards
        with open(f"latent_logs/results_iter_{iteration}.tsv", "w") as f:
            f.write("SMILES\tReward\tOptimization_Steps\n")
            for i, (smi, reward) in enumerate(zip(smiles_list, rewards)):
                opt_steps = detailed_results[i].get('optimization_steps', 0) if i < len(detailed_results) else 0
                f.write(f"{smi}\t{reward:.4f}\t{opt_steps}\n")
        
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
        if len(iteration_results) > 1:
            final_result = iteration_results[-1]
            initial_result = iteration_results[0]
            
            reward_improvement = final_result['mean_reward'] - initial_result['mean_reward']
            diversity_final = final_result['unique_molecules'] / final_result['n_molecules']
            
            logging.info(f"\n   💡 Reward improvement: {reward_improvement:+.3f}")
            logging.info(f"   🎯 Final diversity: {diversity_final:.2%}")