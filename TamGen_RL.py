import os
import time
import numpy as np
import logging
from rdkit import Chem
import torch
from tqdm import tqdm

from fairseq import progress_bar, utils

from utils import prepare_pdb_data, prepare_pdb_data_center, filter_generated_cmpd
from feedback.centroid_optimizer import centroid_shift_optimize

logging.basicConfig(
    filename="latent_logs/debug_latent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TamGenRL:
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
                # On first round, generate using standard TamGen sampling.
                smiles_and_latents = []
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
                print("Preview of valid SMILES:", [s for s, z in smiles_and_latents][:10])
                print("Preview of latent vector shapes:", [z.shape for s, z in smiles_and_latents][:10])
                if len(smiles_and_latents) == 0:
                    logging.error("No valid smiles and latents were generated in iteration 0.")
                    raise RuntimeError("No valid smiles and latents were generated in iteration 0.")
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
                # On subsequent rounds, decode from shifted z_vectors directly
                smiles_decoded = self.generate_from_latents(z_vectors, max_len=128, use_cuda=use_cuda)
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
                noise_sigma=0.05,
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

    def generate_from_latents(self, z_batch, max_len=128, use_cuda=True):
        """
        Generate SMILES from a batch of latent vectors (z_batch) using the decoder and beam search.
        """
        import torch.nn.functional as F

        model = self.models[0]
        model.eval()
        device = next(model.parameters()).device
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=device)
        if z_tensor.ndim == 2:
            z_tensor = z_tensor.unsqueeze(1)  # Add fragment dim

        batch_size = z_tensor.size(0)
        dummy_src = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        dummy_len = torch.full((batch_size,), 5, dtype=torch.long, device=device)

        encoder_out = {
            "encoder_out": None,
            "encoder_padding_mask": None,
            "latent_mean": z_tensor,
            "latent_logstd": None,
        }

        sample = {
            "net_input": {
                "src_tokens": dummy_src,
                "src_lengths": dummy_len,
            },
            "encoder_outs_override": [encoder_out],
        }

        hypos = self.generator.generate(self.models, sample)

        results = []
        for i, hypos_i in enumerate(hypos):
            best_hypo = hypos_i[0]
            hypo_tokens = best_hypo["tokens"].int().cpu()
            hypo_str = self.tgt_dict.string(hypo_tokens, self.args.remove_bpe).strip().replace(" ", "")
            filtered = filter_generated_cmpd(hypo_str)
            if filtered is not None:
                results.append(filtered[0]) 
        return results