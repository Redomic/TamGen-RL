import numpy as np
import torch
from rdkit import Chem
from TamGen_RL import TamGenRL
from utils import prepare_pdb_data
from feedback.centroid_optimizer import centroid_shift_optimize

def validate_iterative_feedback_mechanism():
    """
    Actually test if the feedback loop mechanism works:
    1. Extract latents from iteration 1
    2. Optimize them with centroid shift
    3. Generate molecules from shifted latents
    4. Verify the molecules are different and hopefully better
    """
    print("🔬 FEEDBACK LOOP MECHANISM VALIDATION")
    print("=" * 60)
    
    # Setup
    pdb_id = "3ny8"
    prepare_pdb_data(pdb_id)
    
    demo = TamGenRL(
        data="TamGen_Demo_Data",
        ckpt="checkpoints/crossdocked_model/checkpoint_best.pt",
        use_conditional=True
    )
    demo.reload_data(subset="gen_" + pdb_id.lower())
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("\n🎯 STEP 1: Generate Initial Population")
    # Get initial molecules and latents (exactly like iteration 0)
    initial_smiles, initial_latents, protein_input = extract_baseline_data(demo)
    
    if len(initial_smiles) < 5:
        print(f"❌ Need at least 5 molecules, got {len(initial_smiles)}")
        return False
    
    print(f"   ✅ Generated {len(initial_smiles)} initial molecules")
    print("   📋 Sample molecules:")
    for i, smi in enumerate(initial_smiles[:3]):
        print(f"      {i+1}. {smi}")
    
    print("\n🎯 STEP 2: Optimize Latents (Centroid Shift)")
    # Apply the same optimization as in your actual loop
    z_vectors = np.array(initial_latents)
    docking_scores = [None] * len(initial_smiles)
    
    try:
        z_shifted, rewards, metrics = centroid_shift_optimize(
            z_vectors,
            initial_smiles,
            docking_scores,
            latent_dim=z_vectors.shape[1],
            top_k=min(3, len(initial_smiles)),
            shift_alpha=0.4,  # Use your actual parameters
            lambda_sas=0.3,
            lambda_logp=0.1,
            lambda_mw=0.1,
            noise_sigma=0.05,
        )
        
        shift_magnitude = np.mean(np.linalg.norm(z_shifted - z_vectors, axis=1))
        print(f"   📊 Average latent shift: {shift_magnitude:.4f}")
        print(f"   📊 Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
        return False
    
    print("\n🎯 STEP 3: Generate from Shifted Latents")
    # This is the critical test - do shifted latents produce different molecules?
    shifted_smiles = demo.generate_from_latents_with_protein(
        z_shifted, 
        protein_input, 
        use_cuda=torch.cuda.is_available(),
        batch_size=4
    )
    
    # Filter valid molecules
    shifted_valid = [s for s in shifted_smiles if s and Chem.MolFromSmiles(s) is not None]
    
    print(f"   📊 Generated {len(shifted_valid)}/{len(z_shifted)} valid molecules from shifted latents")
    print("   📋 Sample shifted molecules:")
    for i, smi in enumerate(shifted_valid[:3]):
        print(f"      {i+1}. {smi}")
    
    if len(shifted_valid) == 0:
        print("   ❌ No valid molecules from shifted latents!")
        return False
    
    print("\n🎯 STEP 4: Compare Populations")
    # This is the KEY test - are the populations actually different?
    
    # 4a. Molecular diversity comparison
    initial_set = set(initial_smiles)
    shifted_set = set(shifted_valid)
    
    overlap = len(initial_set.intersection(shifted_set))
    total_unique = len(initial_set.union(shifted_set))
    novelty_rate = len(shifted_set - initial_set) / len(shifted_set) if shifted_set else 0
    
    print(f"   📊 Overlap: {overlap} molecules appear in both populations")
    print(f"   📊 Novelty rate: {novelty_rate:.1%} of shifted molecules are new")
    print(f"   📊 Total unique molecules discovered: {total_unique}")
    
    # 4b. Reward comparison (key test!)
    initial_rewards = rewards[:len(initial_smiles)]  # Original rewards
    
    # Calculate rewards for shifted molecules
    shifted_rewards = []
    for smi in shifted_valid:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            from feedback.reward_utils import compute_reward
            reward, _ = compute_reward(mol, docking_score=None, 
                                     lambda_sas=0.3, lambda_logp=0.1, lambda_mw=0.1)
            shifted_rewards.append(reward)
    
    if shifted_rewards:
        initial_mean = np.mean(initial_rewards)
        shifted_mean = np.mean(shifted_rewards)
        improvement = (shifted_mean - initial_mean) / abs(initial_mean) * 100
        
        print(f"   📊 Initial population mean reward: {initial_mean:.3f}")
        print(f"   📊 Shifted population mean reward: {shifted_mean:.3f}")
        print(f"   📊 Improvement: {improvement:+.1f}%")
        
        # Statistical significance test
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(initial_rewards, shifted_rewards)
            print(f"   📊 Statistical significance: p={p_value:.3f}")
        except:
            print("   ⚠️ Could not compute statistical significance")
    
    print("\n🎯 STEP 5: Validate Mechanism Components")
    
    # 5a. Test if the same latents produce the same molecules (reproducibility)
    print("   🔧 Testing reproducibility...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    test_latent = z_shifted[0:1]  # First shifted latent
    repro_1 = demo._generate_batch_from_latents(test_latent, protein_input, use_cuda=torch.cuda.is_available())
    repro_2 = demo._generate_batch_from_latents(test_latent, protein_input, use_cuda=torch.cuda.is_available())
    
    if repro_1[0] == repro_2[0] and repro_1[0]:
        print("   ✅ Latent injection is deterministic")
    else:
        print("   ⚠️ Latent injection is non-deterministic (might be OK)")
    
    # 5b. Test if different shifted latents produce different molecules
    print("   🔧 Testing latent sensitivity...")
    if len(z_shifted) >= 3:
        test_molecules = []
        for i in range(3):
            mol = demo._generate_batch_from_latents([z_shifted[i]], protein_input, use_cuda=torch.cuda.is_available())
            test_molecules.append(mol[0] if mol and mol[0] else None)
        
        unique_test = len(set([m for m in test_molecules if m]))
        print(f"   📊 {unique_test}/3 different latents produced unique molecules")
        
        if unique_test >= 2:
            print("   ✅ Latent space is responsive to changes")
        else:
            print("   ⚠️ Latent space may not be responsive enough")
    
    print("\n🎯 VALIDATION RESULTS")
    print("=" * 60)
    
    # Overall assessment
    mechanism_working = True
    issues = []
    
    if len(shifted_valid) < len(initial_smiles) * 0.5:
        mechanism_working = False
        issues.append("Low success rate in shifted molecule generation")
    
    if novelty_rate < 0.1:
        mechanism_working = False
        issues.append("Shifted latents produce mostly identical molecules")
    
    if len(shifted_rewards) > 0 and improvement < -5:  # Significant degradation
        mechanism_working = False
        issues.append("Optimization is making molecules significantly worse")
    
    if shift_magnitude < 0.01:
        issues.append("Latent shifts are very small - optimization might be ineffective")
    
    # Print assessment
    if mechanism_working and not issues:
        print("✅ FEEDBACK MECHANISM IS WORKING!")
        print("   - Latent optimization produces meaningful shifts")
        print("   - Shifted latents generate valid, novel molecules")
        if len(shifted_rewards) > 0:
            print(f"   - Population shows {improvement:+.1f}% reward change")
        print("   - Ready for full optimization runs!")
        
    elif mechanism_working:
        print("⚠️ FEEDBACK MECHANISM WORKS BUT HAS ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("   - Consider tuning hyperparameters")
        
    else:
        print("❌ FEEDBACK MECHANISM IS BROKEN:")
        for issue in issues:
            print(f"   - {issue}")
        print("   - Fix these issues before proceeding!")
    
    return mechanism_working and len(issues) == 0

def extract_baseline_data(demo, max_samples=10):
    """Extract baseline molecules and latents like iteration 0"""
    smiles_list = []
    latents_list = []
    protein_input = None
    
    sample_count = 0
    for sample in demo.itr:
        if sample_count >= 2 or len(smiles_list) >= max_samples:
            break
            
        from fairseq import utils
        sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
        
        if protein_input is None:
            protein_input = {
                'src_tokens': sample['net_input']['src_tokens'].clone(),
                'src_lengths': sample['net_input']['src_lengths'].clone(),
                'src_coord': sample['net_input'].get('src_coord', None).clone() if sample['net_input'].get('src_coord') is not None else None,
            }
        
        hypos = demo.task.inference_step(demo.generator, demo.models, sample, None)
        
        for model in demo.models:
            if hasattr(model, 'encoder'):
                batch_size = sample['net_input']['src_tokens'].size(0)
                device = sample['net_input']['src_tokens'].device
                dummy_target = torch.full((batch_size, 5), demo.tgt_dict.bos(), dtype=torch.long, device=device)
                dummy_target[:, -1] = demo.tgt_dict.eos()
                
                encoder_out = model.encoder.forward(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'],
                    src_coord=sample['net_input'].get('src_coord', None),
                    tgt_tokens=dummy_target,
                    tgt_coord=sample['net_input'].get('tgt_coord', None),
                )
                
                if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                    z = encoder_out['latent_mean']
                    z_np = z.detach().cpu().numpy()
                    if z_np.ndim == 3:
                        z_np = z_np.mean(axis=1)
                    
                    for i, sample_id in enumerate(sample['id'].tolist()):
                        if i < len(hypos) and len(hypos[i]) > 0:
                            best_hypo = hypos[i][0]
                            hypo_tokens = best_hypo["tokens"].int().cpu()
                            hypo_str = demo.tgt_dict.string(hypo_tokens, demo.args.remove_bpe).strip().replace(" ", "")
                            
                            mol = Chem.MolFromSmiles(hypo_str)
                            if mol is not None and len(smiles_list) < max_samples:
                                smiles_list.append(hypo_str)
                                latents_list.append(z_np[i])
        
        sample_count += 1
    
    return smiles_list, latents_list, protein_input

if __name__ == "__main__":
    success = validate_iterative_feedback_mechanism()
    if success:
        print("\n🚀 Your feedback loop mechanism is validated and ready!")
    else:
        print("\n🔧 Fix the feedback mechanism before running full optimization!")