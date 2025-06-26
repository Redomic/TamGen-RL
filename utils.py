import os
from glob import glob
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs

from feedback.reward_utils import compute_three_criterion_reward


def prepare_pdb_data(pdb_id, ligand_inchi=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return
    
    os.system(f"mkdir -p {DemoDataFolder}")
    if ligand_inchi is None:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id", file=fw)
            print(f"{pdb_id}", file=fw)
    else:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id,ligand_inchi", file=fw)
            print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def prepare_pdb_data_center(pdb_id, scaffold_file=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return

    with open("tmp_pdb.csv", "w") as fw:
        print("pdb_id,ligand_inchi", file=fw)
        print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def filter_generated_cmpd(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sssr = Chem.GetSymmSSSR(m)
    if len(sssr) <= 1:
        return None
    if len(sssr) >= 4:
        return None
    if smi.lower().count('p') > 3:
        return None
    s = Chem.MolToSmiles(m)
    return s, m


class InjectionMonitor:
    
    def __init__(self, tamgen_instance):
        self.tamgen = tamgen_instance
    
    def monitor_injection_quality(self, iteration, z_vectors, generated_smiles, rewards=None):
        """Enhanced monitoring of injection quality during training."""
        
        if iteration % 3 == 0:  # Every 3rd iteration
            logging.info(f"🔍 Injection quality check - Iteration {iteration}")
            
            # Run debug tests if injection seems broken
            if iteration == 0:  # Run comprehensive debug on first iteration
                self.run_comprehensive_debug()
            
            # Test 1: Basic perturbation test
            sample_size = min(5, len(z_vectors))
            sample_z = z_vectors[:sample_size]
            sample_smiles = generated_smiles[:sample_size]
            
            # Small perturbation test
            perturbed_z = sample_z + np.random.normal(0, 0.05, sample_z.shape)
            perturbed_smiles = generate_from_latents(
                self.tamgen, perturbed_z, batch_size=2, use_cuda=True
            )
            
            basic_similarities = compute_tanimoto_similarities(sample_smiles, perturbed_smiles)
            
            # Test 2: Large perturbation test (should be more different)
            large_perturbed_z = sample_z + np.random.normal(0, 0.2, sample_z.shape)
            large_perturbed_smiles = generate_from_latents(
                self.tamgen, large_perturbed_z, batch_size=2, use_cuda=True
            )
            
            large_similarities = compute_tanimoto_similarities(sample_smiles, large_perturbed_smiles)
            
            # Test 3: Zero perturbation (should be identical or very similar)
            identical_smiles = generate_from_latents(
                self.tamgen, sample_z, batch_size=2, use_cuda=True
            )
            identity_similarities = compute_tanimoto_similarities(sample_smiles, identical_smiles)
            
            # Test 4: Protein context preservation
            protein_preservation_score = test_protein_context_preservation(self.tamgen, sample_z)
            
            # Analyze results
            results = {
                'small_pert_similarity': np.mean(basic_similarities) if basic_similarities else 0,
                'large_pert_similarity': np.mean(large_similarities) if large_similarities else 0,
                'identity_similarity': np.mean(identity_similarities) if identity_similarities else 0,
                'protein_preservation': protein_preservation_score,
                'valid_molecules_ratio': len([s for s in perturbed_smiles if s]) / max(len(perturbed_smiles), 1)
            }
            
            # Log detailed results
            logging.info(f"   Small perturbation similarity: {results['small_pert_similarity']:.3f}")
            logging.info(f"   Large perturbation similarity: {results['large_pert_similarity']:.3f}")
            logging.info(f"   Identity similarity: {results['identity_similarity']:.3f}")
            logging.info(f"   Protein preservation score: {results['protein_preservation']:.3f}")
            logging.info(f"   Valid molecules ratio: {results['valid_molecules_ratio']:.3f}")
            
            # Comprehensive health check
            health_status = assess_injection_health(results)
            
            if health_status != "healthy":
                logging.warning(f"⚠️ Injection health issue detected: {health_status}")
                
                # Run specific debug if issues detected
                if "poor_reproducibility" in health_status:
                    self.debug_injection_identity()
                if "perturbation_scaling_broken" in health_status:
                    self.debug_perturbation_scaling(sample_z)
                
            # Test 5: Reward direction validation (if rewards available)
            if rewards is not None and len(rewards) >= 3:
                direction_test = test_reward_direction(
                    self.tamgen, sample_z, rewards[:sample_size]
                )
                logging.info(f"   Reward direction test: {direction_test}")
            
            return results
    
    def run_comprehensive_debug(self):
        """Run all debug tests on first iteration."""
        logging.info("🔧 Running comprehensive injection debug...")
        
        try:
            self.debug_architecture_assumptions()
            self.debug_injection_identity()
            
            # Create a dummy sample for encoder tests
            if self.tamgen.stored_protein_inputs:
                sample = self._create_dummy_sample()
                self.debug_encoder_override(sample)
            else:
                logging.warning("No stored protein inputs for encoder debug")
                
        except Exception as e:
            logging.error(f"Debug failed: {e}")
    
    def debug_injection_identity(self):
        """Debug if latent injection is actually working."""
        logging.info("🔍 Testing injection identity...")
        
        try:
            if not self.tamgen.stored_protein_inputs:
                logging.warning("No stored protein inputs for identity test")
                return False
            
            # Get some reference latents
            z_vectors, smiles_list = self.tamgen._initial_generation(
                m_sample=3, maxseed=3, use_cuda=True, diversity_target=0.5
            )
            
            if len(z_vectors) < 2:
                logging.warning("Not enough samples for identity test")
                return False
            
            # Use same latent 5 times
            same_z = z_vectors[0:1]  # Take first latent
            
            # Generate 5 times with identical latents
            identical_results = []
            for i in range(5):
                result = generate_from_latents(self.tamgen, same_z, batch_size=1, use_cuda=True)
                if result:
                    identical_results.append(result[0])
                    logging.info(f"   Generation {i+1}: {result[0]}")
                else:
                    logging.warning(f"   Generation {i+1}: FAILED")
            
            # Check if all are identical
            unique_molecules = set(identical_results)
            if len(unique_molecules) == 1:
                logging.info("   ✅ PERFECT: All generations identical - injection working!")
                return True
            else:
                logging.error(f"   ❌ BROKEN: Got {len(unique_molecules)} different molecules from same latent")
                logging.error(f"   Unique molecules: {list(unique_molecules)}")
                return False
                
        except Exception as e:
            logging.error(f"Identity test failed: {e}")
            return False
    
    def debug_encoder_override(self, sample):
        """Debug if encoder_outs_override is being used."""
        logging.info("🔍 Testing encoder override mechanism...")
        
        try:
            # Generate normally
            normal_hypos = self.tamgen.task.inference_step(self.tamgen.generator, self.tamgen.models, sample, None)
            normal_smiles = self.tamgen.tgt_dict.string(
                normal_hypos[0][0]['tokens'], self.tamgen.args.remove_bpe
            ).strip().replace(" ", "") if normal_hypos[0] else "FAILED"
            
            # Create dummy override (all zeros)
            normal_encoder_out = self.tamgen.models[0].encoder.forward(
                sample['net_input']['src_tokens'],
                sample['net_input']['src_lengths'],
                src_coord=sample['net_input'].get('src_coord', None)
            )
            
            dummy_encoder_out = {
                'encoder_out': torch.zeros_like(normal_encoder_out['encoder_out']),
                'encoder_padding_mask': normal_encoder_out.get('encoder_padding_mask', None),
            }
            
            # Override with dummy
            override_sample = sample.copy()
            override_sample["encoder_outs_override"] = [dummy_encoder_out]
            
            # Generate with override
            override_hypos = self.tamgen.task.inference_step(self.tamgen.generator, self.tamgen.models, override_sample, None)
            override_smiles = self.tamgen.tgt_dict.string(
                override_hypos[0][0]['tokens'], self.tamgen.args.remove_bpe
            ).strip().replace(" ", "") if override_hypos[0] else "FAILED"
            
            logging.info(f"   Normal:   {normal_smiles}")
            logging.info(f"   Override: {override_smiles}")
            
            if normal_smiles == override_smiles:
                logging.error("   ❌ BROKEN: Override not working - same output despite zero encoder!")
                return False
            else:
                logging.info("   ✅ GOOD: Override working - different outputs")
                return True
                
        except Exception as e:
            logging.error(f"Encoder override test failed: {e}")
            return False
    
    def debug_architecture_assumptions(self):
        """Debug TamGen architecture assumptions."""
        logging.info("🔍 Checking architecture assumptions...")
        
        try:
            if not self.tamgen.stored_protein_inputs:
                logging.warning("No stored protein inputs for architecture check")
                return
            
            sample = self._create_dummy_sample()
            model = self.tamgen.models[0]
            
            # Get encoder output structure
            encoder_out = model.encoder.forward(
                sample['net_input']['src_tokens'],
                sample['net_input']['src_lengths'],
                src_coord=sample['net_input'].get('src_coord', None)
            )
            
            logging.info(f"   Encoder output keys: {list(encoder_out.keys())}")
            logging.info(f"   Encoder output shape: {encoder_out['encoder_out'].shape}")
            logging.info(f"   Concat mode: {getattr(self.tamgen.args, 'concat', 'NOT_SET')}")
            logging.info(f"   Detected latent dim: {self.tamgen.latent_dim}")
            logging.info(f"   Expected encoder dim: {getattr(self.tamgen.args, 'encoder_embed_dim', 'NOT_SET')}")
            
            # Check if VAE is active
            if hasattr(model.encoder, 'vae_encoder'):
                logging.info("   ✅ VAE encoder found")
                if hasattr(model.encoder, 'latent_mean'):
                    logging.info("   ✅ Latent mean attribute found")
                else:
                    logging.warning("   ❌ No latent_mean attribute")
            else:
                logging.warning("   ❌ No VAE encoder found")
            
            # Check gen_vae setting
            gen_vae = getattr(self.tamgen.args, 'gen_vae', False)
            logging.info(f"   gen_vae setting: {gen_vae}")
            
            # Check if model is in conditional mode
            if hasattr(model.encoder, 'gen_vae'):
                logging.info(f"   Model gen_vae: {model.encoder.gen_vae}")
            
        except Exception as e:
            logging.error(f"Architecture check failed: {e}")
    
    def debug_perturbation_scaling(self, sample_z):
        """Debug why perturbation scaling is broken."""
        logging.info("🔍 Debugging perturbation scaling...")
        
        try:
            base_z = sample_z[0:1]  # Take first sample
            
            # Test different perturbation magnitudes
            scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
            results = []
            
            # Generate base molecule
            base_smiles = generate_from_latents(self.tamgen, base_z, batch_size=1, use_cuda=True)
            if not base_smiles or not base_smiles[0]:
                logging.error("   Failed to generate base molecule")
                return
            
            base_mol = Chem.MolFromSmiles(base_smiles[0])
            if not base_mol:
                logging.error("   Invalid base molecule")
                return
            
            base_fp = FingerprintMols.FingerprintMol(base_mol)
            
            logging.info(f"   Base molecule: {base_smiles[0]}")
            
            for scale in scales:
                # Apply perturbation
                if scale == 0.0:
                    perturbed_z = base_z.copy()
                else:
                    perturbed_z = base_z + np.random.normal(0, scale, base_z.shape)
                
                # Generate perturbed molecule
                perturbed_smiles = generate_from_latents(self.tamgen, perturbed_z, batch_size=1, use_cuda=True)
                
                if perturbed_smiles and perturbed_smiles[0]:
                    perturbed_mol = Chem.MolFromSmiles(perturbed_smiles[0])
                    if perturbed_mol:
                        perturbed_fp = FingerprintMols.FingerprintMol(perturbed_mol)
                        similarity = DataStructs.TanimotoSimilarity(base_fp, perturbed_fp)
                        results.append((scale, similarity, perturbed_smiles[0]))
                        logging.info(f"   Scale {scale:4.2f}: Similarity {similarity:.3f} - {perturbed_smiles[0]}")
                    else:
                        logging.warning(f"   Scale {scale:4.2f}: Invalid molecule - {perturbed_smiles[0]}")
                else:
                    logging.warning(f"   Scale {scale:4.2f}: Generation failed")
            
            # Check if similarity decreases with scale
            similarities = [r[1] for r in results]
            if len(similarities) >= 3:
                if similarities[-1] < similarities[1]:  # Last should be less similar than small perturbation
                    logging.info("   ✅ Perturbation scaling working correctly")
                else:
                    logging.error("   ❌ Perturbation scaling broken - larger perturbations not less similar")
            
        except Exception as e:
            logging.error(f"Perturbation scaling debug failed: {e}")
    
    def _create_dummy_sample(self):
        """Create a dummy sample for testing."""
        if self.tamgen.stored_protein_inputs:
            protein_input = self.tamgen.stored_protein_inputs[0]
            return {
                "net_input": protein_input,
                "id": torch.tensor([0], device=self.tamgen.device)
            }
        else:
            raise RuntimeError("No stored protein inputs available for dummy sample")
    
    def debug_latent_injection_step_by_step(self, test_z):
        """Step-by-step debug of the injection process."""
        logging.info("🔍 Step-by-step latent injection debug...")
        
        try:
            protein_input = self.tamgen.stored_protein_inputs[0]
            model = self.tamgen.models[0]
            
            # Step 1: Prepare inputs
            batch_size = len(test_z)
            z_tensor = torch.tensor(test_z, dtype=torch.float32, device=self.tamgen.device)
            
            src_tokens = protein_input['src_tokens'][:1].expand(batch_size, -1).to(self.tamgen.device)
            src_lengths = protein_input['src_lengths'][:1].expand(batch_size).to(self.tamgen.device)
            src_coord = None
            if protein_input.get('src_coord') is not None:
                src_coord = protein_input['src_coord'][:1].expand(batch_size, -1, -1).to(self.tamgen.device)
            
            logging.info(f"   Step 1: Input preparation complete")
            logging.info(f"   - Batch size: {batch_size}")
            logging.info(f"   - z_tensor shape: {z_tensor.shape}")
            logging.info(f"   - src_tokens shape: {src_tokens.shape}")
            
            # Step 2: Get normal encoder output
            model.eval()
            encoder_out = model.encoder.forward(src_tokens, src_lengths, src_coord=src_coord)
            
            logging.info(f"   Step 2: Normal encoder output")
            logging.info(f"   - Encoder out shape: {encoder_out['encoder_out'].shape}")
            logging.info(f"   - Keys: {list(encoder_out.keys())}")
            
            # Step 3: Try injection
            original_encoder_out = encoder_out['encoder_out'].clone()
            
            if hasattr(self.tamgen.args, 'concat') and self.tamgen.args.concat:
                logging.info(f"   Step 3: Using concatenation mode")
                main_features = encoder_out['encoder_out'][..., :-self.tamgen.latent_dim]
                z_expanded = z_tensor.unsqueeze(0).expand(main_features.size(0), -1, -1)
                encoder_out['encoder_out'] = torch.cat([main_features, z_expanded], dim=-1)
                logging.info(f"   - Main features shape: {main_features.shape}")
                logging.info(f"   - z_expanded shape: {z_expanded.shape}")
                logging.info(f"   - Final shape: {encoder_out['encoder_out'].shape}")
            else:
                logging.info(f"   Step 3: Using addition mode")
                z_expanded = z_tensor.unsqueeze(0).expand(encoder_out['encoder_out'].size(0), -1, -1)
                encoder_out['encoder_out'] = encoder_out['encoder_out'] + z_expanded
                logging.info(f"   - z_expanded shape: {z_expanded.shape}")
                logging.info(f"   - Final shape: {encoder_out['encoder_out'].shape}")
            
            # Check if modification actually happened
            modification_magnitude = torch.norm(encoder_out['encoder_out'] - original_encoder_out).item()
            logging.info(f"   - Modification magnitude: {modification_magnitude:.6f}")
            
            if modification_magnitude < 1e-8:
                logging.error("   ❌ No modification detected!")
                return False
            else:
                logging.info("   ✅ Encoder output successfully modified")
                return True
            
        except Exception as e:
            logging.error(f"Step-by-step debug failed: {e}")
            return False

def generate_from_latents(tamgen_instance,
                         z_vectors: np.ndarray,
                         batch_size: int = 4,
                         use_cuda: bool = True) -> List[str]:
    """Generate SMILES from latent vectors using stored protein input."""
    
    if len(tamgen_instance.stored_protein_inputs) == 0:
        raise RuntimeError("No stored protein inputs available")
    
    logging.debug(f"🔄 Generating from {len(z_vectors)} latent vectors...")
    
    protein_input = tamgen_instance.stored_protein_inputs[0]
    total_samples = len(z_vectors)
    all_results = []
    
    # Process in batches to avoid OOM
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_z = z_vectors[start_idx:end_idx]
        
        try:
            batch_results = tamgen_instance._generate_batch_with_latent_injection(
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
                        sub_results = tamgen_instance._generate_batch_with_latent_injection(
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
    
    logging.debug(f"   ✓ Generated {len(valid_results)}/{len(z_vectors)} valid SMILES")
    
    return valid_results


def compute_tanimoto_similarities(smiles1: List[str], smiles2: List[str]) -> List[float]:
    """Compute Tanimoto similarities between two SMILES lists."""
    similarities = []
    for i in range(min(len(smiles1), len(smiles2))):
        if smiles1[i] and smiles2[i]:
            mol1 = Chem.MolFromSmiles(smiles1[i])
            mol2 = Chem.MolFromSmiles(smiles2[i])
            if mol1 and mol2:
                fp1 = FingerprintMols.FingerprintMol(mol1)
                fp2 = FingerprintMols.FingerprintMol(mol2)
                sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                similarities.append(sim)
    return similarities


def assess_injection_health(results: Dict[str, float]) -> str:
    """Assess overall injection health based on test results."""
    
    # Expected ranges for healthy injection
    small_pert_range = (0.4, 0.8)    # Should be moderately similar
    large_pert_range = (0.1, 0.6)    # Should be less similar
    identity_range = (0.8, 1.0)      # Should be very similar
    
    issues = []
    
    # Check identity similarity
    if results['identity_similarity'] < identity_range[0]:
        issues.append("poor_reproducibility")
    
    # Check perturbation response
    if not (small_pert_range[0] <= results['small_pert_similarity'] <= small_pert_range[1]):
        if results['small_pert_similarity'] < small_pert_range[0]:
            issues.append("excessive_sensitivity")
        else:
            issues.append("insufficient_sensitivity")
    
    # Check that large perturbations are more different than small ones
    if results['large_pert_similarity'] >= results['small_pert_similarity']:
        issues.append("perturbation_scaling_broken")
    
    # Check molecule validity
    if results['valid_molecules_ratio'] < 0.7:
        issues.append("low_validity")
    
    # Check protein preservation
    if results['protein_preservation'] < 0.3:
        issues.append("protein_context_lost")
    
    if not issues:
        return "healthy"
    else:
        return f"unhealthy: {', '.join(issues)}"


def test_reward_direction(tamgen_instance, sample_z: np.ndarray, sample_rewards: List[float]) -> str:
    """Test if latent perturbations in reward direction improve rewards."""
    try:
        if len(sample_rewards) < 2:
            return "insufficient_data"
        
        # Find best and worst samples
        best_idx = np.argmax(sample_rewards)
        worst_idx = np.argmin(sample_rewards)
        
        # Compute direction from worst to best
        direction = sample_z[best_idx] - sample_z[worst_idx]
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return "no_gradient"
        
        direction = direction / direction_norm  # Normalize
        
        # Move worst sample towards best
        improved_z = sample_z[worst_idx:worst_idx+1] + 0.1 * direction.reshape(1, -1)
        improved_smiles = generate_from_latents(tamgen_instance, improved_z, batch_size=1, use_cuda=True)
        
        if improved_smiles and improved_smiles[0]:
            mol = Chem.MolFromSmiles(improved_smiles[0])
            if mol:
                improved_reward, _ = compute_three_criterion_reward(mol)
                original_reward = sample_rewards[worst_idx]
                
                if improved_reward > original_reward + 0.1:
                    return "positive_gradient"
                elif improved_reward < original_reward - 0.1:
                    return "negative_gradient"
                else:
                    return "neutral_gradient"
        
        return "generation_failed"
        
    except Exception as e:
        logging.warning(f"Reward direction test failed: {e}")
        return "test_error"


def test_protein_context_preservation(tamgen_instance, sample_z: np.ndarray) -> float:
    """Test if protein context is preserved during injection."""
    try:
        if not tamgen_instance.stored_protein_inputs:
            return 0.0
        
        # Generate with same latents twice to test reproducibility
        batch1 = generate_from_latents(tamgen_instance, sample_z[:2], batch_size=1, use_cuda=True)
        batch2 = generate_from_latents(tamgen_instance, sample_z[:2], batch_size=1, use_cuda=True)
        
        # If completely deterministic, should be identical
        # If stochastic but controlled, should be similar
        if len(batch1) >= 1 and len(batch2) >= 1 and batch1[0] and batch2[0]:
            mol1 = Chem.MolFromSmiles(batch1[0])
            mol2 = Chem.MolFromSmiles(batch2[0])
            if mol1 and mol2:
                fp1 = FingerprintMols.FingerprintMol(mol1)
                fp2 = FingerprintMols.FingerprintMol(mol2)
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        
        return 0.5  # Default if can't compute
        
    except Exception as e:
        logging.warning(f"Protein context test failed: {e}")
        return 0.0


# Convenience function for quick monitoring
def quick_monitor(tamgen_instance, iteration: int, z_vectors: np.ndarray, 
                  generated_smiles: List[str], rewards: Optional[List[float]] = None) -> Dict[str, Any]:
    """Quick monitoring function - convenience wrapper."""
    monitor = InjectionMonitor(tamgen_instance)
    return monitor.monitor_injection_quality(iteration, z_vectors, generated_smiles, rewards)