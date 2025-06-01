

# centroid_shift_optimizer.py

import numpy as np
from reward_utils_praneeth import compute_advanced_reward, compute_diversity_bonus
from reward_model_praneeth import LatentRewardModel

class CentroidShiftOptimizer:
    def __init__(self, decoder, reward_fn=compute_advanced_reward):
        self.decoder = decoder
        self.reward_fn = reward_fn
        self.reward_model = LatentRewardModel()

    def optimize(self, initial_z, steps=50, use_gradient=True):
        z = initial_z.copy()
        best_reward = -np.inf
        best_smiles = None
        history = []

        for step in range(steps):
            smiles = self.decoder(z)
            reward = self.reward_fn(smiles)
            self.reward_model.add(z, reward)

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

            # Add diversity bonus using perturbations
            decoded_smiles = [self.decoder(z + np.random.randn(*z.shape) * 0.01) for _ in range(3)]
            reward += compute_diversity_bonus(decoded_smiles)

            # Compute shift
            if use_gradient:
                shift = self.reward_model.get_gradient_based_shift(z)
            else:
                shift = self.reward_model.get_centroid_shift()
                if shift is None:
                    shift = np.random.randn(*z.shape)
                shift = shift - z
                shift = shift / (np.linalg.norm(shift) + 1e-8)

            z += 0.2 * shift  # learning rate / step size

            if step % 5 == 0:
                self.reward_model.train()

            history.append((step, reward, smiles))

        return best_smiles, best_reward, history






