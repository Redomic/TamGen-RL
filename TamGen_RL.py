import os
import time
import numpy as np
import logging
from rdkit import Chem
import torch
from tqdm import tqdm

from TamGen_Demo import TamGenDemo
from fairseq import progress_bar, utils

from utils import prepare_pdb_data, prepare_pdb_data_center, filter_generated_cmpd
from feedback.centroid_optimizer import centroid_shift_optimize

logging.basicConfig(
    filename="latent_logs/debug_latent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TamGenRL(TamGenDemo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_protein_inputs = None  # Store original protein inputs for reuse
        
    def sample(
        self,
        m_sample=100,
        num_iter=5,
        latent_dim=256,
        alpha=0.5,
        top_k=50,
        lambda_sas=0.3,
        lambda_logp=0.1,
        lambda_mw=0.1,
        maxseed=20,
        use_cuda=True,
        **kwargs
    ):
        z_vectors = None
        smiles_list = None

        os.makedirs("latent_logs", exist_ok=True)
        logging.info("🚀 Feedback loop started.")

        # Prepare dataset and model as usual
        print("⚙️  Starting closed-loop optimization...")

        for iteration in range(num_iter):
            print(f"\n🚀 Iteration {iteration + 1}/{num_iter}")
            start_time = time.time()

            if iteration == 0:
                # On first round, generate using standard TamGen sampling and store protein inputs
                smiles_and_latents = []
                self.stored_protein_inputs = []  # Initialize storage for protein inputs
                
                for seed in tqdm(range(1, maxseed), total=maxseed):
                    if len(smiles_and_latents) >= m_sample:
                        break
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    with progress_bar.build_progress_bar(self.args, self.itr) as t:
                        for sample in t:
                            sample = utils.move_to_cuda(sample) if use_cuda else sample
                            if 'net_input' not in sample:
                                continue
                            
                            # Store protein input for later reuse
                            protein_input = {
                                'src_tokens': sample['net_input']['src_tokens'].clone(),
                                'src_lengths': sample['net_input']['src_lengths'].clone(),
                                'src_coord': sample['net_input'].get('src_coord', None).clone() if sample['net_input'].get('src_coord') is not None else None,
                            }
                            if len(self.stored_protein_inputs) == 0:  # Store first valid protein input
                                self.stored_protein_inputs.append(protein_input)
                                print(f"📦 Stored protein input with shape: {protein_input['src_tokens'].shape}")
                            
                            prefix_tokens = None
                            if use_cuda:
                                with torch.no_grad():
                                    hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
                            else:
                                hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
                            
                            for model in self.models:
                                if hasattr(model, 'encoder'):
                                    if use_cuda:
                                        with torch.no_grad():
                                            encoder_out = model.encoder.forward(
                                                sample['net_input']['src_tokens'],
                                                sample['net_input']['src_lengths'],
                                                src_coord=sample['net_input'].get('src_coord', None),
                                                tgt_tokens=sample.get('target', None),
                                                tgt_coord=sample['net_input'].get('tgt_coord', None),
                                            )
                                    else:
                                        encoder_out = model.encoder.forward(
                                            sample['net_input']['src_tokens'],
                                            sample['net_input']['src_lengths'],
                                            src_coord=sample['net_input'].get('src_coord', None),
                                            tgt_tokens=sample.get('target', None),
                                            tgt_coord=sample['net_input'].get('tgt_coord', None),
                                        )
                                    
                                    if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                                        z = encoder_out['latent_mean']
                                        z_np = z.detach().cpu().numpy()
                                        if z_np.ndim == 3:
                                            z_np = z_np.mean(axis=1)
                                        assert z_np.ndim == 2, f"Expected 2D array, got {z_np.shape}"
                                        for i, sample_id in enumerate(sample['id'].tolist()):
                                            has_target = sample['target'] is not None
                                            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], self.tgt_dict.pad())
                                            target_tokens = None
                                            if has_target:
                                                target_tokens = utils.strip_pad(sample['target'][i, :], self.tgt_dict.pad()).int().cpu()
                                            src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                                            target_str = self.tgt_dict.string(target_tokens, self.args.remove_bpe, escape_unk=True)
                                            tmps = target_str.strip().replace(" ", "")
                                            mol = Chem.MolFromSmiles(tmps)
                                            if mol is not None:
                                                smiles_and_latents.append((tmps, z_np[i]))
                                            if len(smiles_and_latents) >= m_sample:
                                                break
                                    if len(smiles_and_latents) >= m_sample:
                                        break
                            if len(smiles_and_latents) >= m_sample:
                                break
                    if len(smiles_and_latents) >= m_sample:
                        break
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        
                print(f"Total valid SMILES and latents after all seeds: {len(smiles_and_latents)}")
                
                if len(smiles_and_latents) == 0:
                    logging.error("No valid smiles and latents were generated in iteration 0.")
                    raise RuntimeError("No valid smiles and latents were generated in iteration 0.")
                
                if len(self.stored_protein_inputs) == 0:
                    logging.error("No protein inputs were stored.")
                    raise RuntimeError("No protein inputs were stored.")
                
                # Reconstruct lists to ensure 1:1 mapping
                smiles_list = [s for s, z in smiles_and_latents]
                unique_smiles_count = len(set(smiles_list))
                total_smiles_count = len(smiles_list)
                if unique_smiles_count < total_smiles_count:
                    logging.warning(f"Duplicate SMILES detected: {total_smiles_count - unique_smiles_count} duplicates.")
                print(f"Unique SMILES this iteration: {unique_smiles_count} / {total_smiles_count}")
                z_vectors = np.stack([z for s, z in smiles_and_latents])

                if z_vectors is None or smiles_list is None:
                    raise RuntimeError("No valid molecules generated in the first iteration, cannot proceed with feedback loop.")

            else:
                if z_vectors is None or smiles_list is None:
                    raise RuntimeError("Latent vectors or SMILES not initialized in the first iteration.")
                if self.stored_protein_inputs is None or len(self.stored_protein_inputs) == 0:
                    raise RuntimeError("No stored protein inputs available for subsequent iterations.")
                
                # On subsequent rounds, decode from shifted z_vectors using stored protein input
                print(f"🔄 Generating from {len(z_vectors)} shifted latent vectors...")
                smiles_decoded = self.generate_from_latents_with_protein(
                    z_vectors, 
                    self.stored_protein_inputs[0], 
                    max_len=128, 
                    use_cuda=use_cuda,
                    batch_size=4  # Start with small batch size
                )
                
                # Keep only non-empty SMILES and corresponding latents
                smiles_and_latents = [(s, z) for s, z in zip(smiles_decoded, z_vectors) if s]
                if len(smiles_and_latents) == 0:
                    logging.error("No valid SMILES/latents in this iteration after decoding.")
                    raise RuntimeError("No valid SMILES/latents in this iteration after decoding.")
                
                smiles_list = [s for s, z in smiles_and_latents]
                z_vectors = np.stack([z for s, z in smiles_and_latents])
                unique_smiles_count = len(set(smiles_list))
                total_smiles_count = len(smiles_list)
                if unique_smiles_count < total_smiles_count:
                    logging.warning(f"Duplicate SMILES detected: {total_smiles_count - unique_smiles_count} duplicates.")
                print(f"Unique SMILES this iteration: {unique_smiles_count} / {total_smiles_count}")

            # 3. Placeholder Docking Scores (not used)
            docking_scores = [None] * len(smiles_list)

            print("📊 Optimizing latent space...")
            z_shifted, rewards, metrics = centroid_shift_optimize(
                z_vectors,
                smiles_list,
                docking_scores,
                latent_dim=latent_dim,
                top_k=top_k,
                shift_alpha=alpha,
                lambda_sas=lambda_sas,
                lambda_logp=lambda_logp,
                lambda_mw=lambda_mw,
                noise_sigma=0.1,
            )
            print("   ✔ Optimization complete.")

            # 5. Save Outputs
            print("💾 Saving latent vectors and rewards...")
            np.savetxt("latent_logs/latent_vectors.tsv", np.array(z_shifted), fmt="%.5f")
            with open(f"latent_logs/rewards_iter_{iteration + 1}.tsv", "w") as f:
                for smi, r in zip(smiles_list, rewards):
                    f.write(f"{smi}\t{r:.4f}\n")

            logging.info(f"✅ Completed Iteration {iteration + 1} in {time.time() - start_time:.2f}s")
            print(f"✅ Iteration {iteration + 1} complete.")

            z_vectors = z_shifted
            # Aggressively delete big objects and clear CUDA memory
            del z_shifted, rewards, metrics, docking_scores
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Clear CUDA cache to help avoid OOM
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        print("\n🎉 Feedback loop finished. Ready for SGDS optimization.")
        return smiles_list
    
    def generate_from_latents_with_protein(self, z_batch, protein_input, max_len=128, use_cuda=True, batch_size=8):
        """
        Generate SMILES using shifted latents with original protein structure.
        Process in smaller batches to avoid OOM.
        """
        # Fix the numpy array warning
        if isinstance(z_batch, list):
            z_batch = np.array(z_batch)
        
        total_samples = len(z_batch)
        all_results = []
        
        # Process in batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_z = z_batch[start_idx:end_idx]
            
            try:
                batch_results = self._generate_batch_from_latents(batch_z, protein_input, max_len, use_cuda)
                all_results.extend(batch_results)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ OOM in batch {start_idx//batch_size + 1}, trying smaller batch size")
                    # Try with even smaller batch size
                    smaller_batch_size = max(1, batch_size // 2)
                    for sub_start in range(start_idx, end_idx, smaller_batch_size):
                        sub_end = min(sub_start + smaller_batch_size, end_idx)
                        sub_batch_z = z_batch[sub_start:sub_end]
                        try:
                            sub_results = self._generate_batch_from_latents(sub_batch_z, protein_input, max_len, use_cuda)
                            all_results.extend(sub_results)
                        except RuntimeError:
                            print(f"❌ Failed to process samples {sub_start}-{sub_end}, adding empty results")
                            all_results.extend([""] * (sub_end - sub_start))
                else:
                    raise e
            
            # Clear cache after each batch
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        return all_results

    def _generate_batch_from_latents(self, z_batch, protein_input, max_len=128, use_cuda=True):
        """
        Generate using TamGen's proper pipeline with injected latent vectors.
        """
        model = self.models[0]
        model.eval()
        device = next(model.parameters()).device
        
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=device)
        batch_size = z_tensor.size(0)
        
        # Expand protein input to match batch size
        src_tokens = protein_input['src_tokens'][:1].expand(batch_size, -1).to(device)
        src_lengths = protein_input['src_lengths'][:1].expand(batch_size).to(device)
        src_coord = None
        if protein_input.get('src_coord') is not None:
            src_coord = protein_input['src_coord'][:1].expand(batch_size, -1, -1).to(device)
        
        # Create dummy target tokens (required for VAE encoder)
        # Use the BOS token repeated to create a minimal target
        dummy_target = torch.full((batch_size, 5), self.tgt_dict.bos(), dtype=torch.long, device=device)
        dummy_target[:, -1] = self.tgt_dict.eos()  # End with EOS
        
        # Create sample exactly like in the first iteration
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_coord": src_coord,
            },
            "target": dummy_target,  # This is crucial for VAE encoder
            "id": torch.arange(batch_size, device=device),
        }
        
        # Store original latent values in the encoder
        original_latent_mean = None
        original_latent_logstd = None
        
        # Hook into the encoder to replace latent values
        def encoder_hook(module, input, output):
            nonlocal original_latent_mean, original_latent_logstd
            if isinstance(output, dict) and 'latent_mean' in output:
                original_latent_mean = output['latent_mean']
                original_latent_logstd = output['latent_logstd']
                
                # Replace with our shifted latent vectors
                # Ensure the shape matches the expected format
                if z_tensor.ndim == 2:
                    # z_tensor is (batch_size, latent_dim)
                    # TamGen expects (seq_len, batch_size, latent_dim)
                    seq_len = output['latent_mean'].size(0) if output['latent_mean'] is not None else 1
                    replacement_latent = z_tensor.unsqueeze(0).expand(seq_len, -1, -1)
                else:
                    replacement_latent = z_tensor
                
                output['latent_mean'] = replacement_latent
                output['latent_logstd'] = torch.zeros_like(replacement_latent)
            
            return output
        
        # Register the hook
        hook_handle = model.encoder.register_forward_hook(encoder_hook)
        
        try:
            # Use the normal TamGen inference pipeline
            prefix_tokens = None
            hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
                    
        finally:
            # Always remove the hook
            hook_handle.remove()
            
            # Restore original values if needed
            if hasattr(model.encoder, 'latent_mean'):
                model.encoder.latent_mean = original_latent_mean
                model.encoder.latent_logstd = original_latent_logstd
        
        # Process results
        results = []
        for i, hypos_i in enumerate(hypos):
            if len(hypos_i) > 0:
                best_hypo = hypos_i[0]
                hypo_tokens = best_hypo["tokens"].int().cpu()
                hypo_str = self.tgt_dict.string(hypo_tokens, self.args.remove_bpe).strip().replace(" ", "")
                
                # Accept all valid RDKit molecules
                from rdkit import Chem
                mol = Chem.MolFromSmiles(hypo_str)
                if mol is not None:
                    results.append(hypo_str)
                else:
                    results.append("")
            else:
                results.append("")
        
        return results