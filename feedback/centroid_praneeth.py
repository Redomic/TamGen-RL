class CentroidShiftOptimizer:
    def __init__(self, decoder, reward_fn=compute_advanced_reward):
        self.decoder = decoder
        self.reward_fn = reward_fn
        self.reward_model = LatentRewardModel()

    def optimize(self, initial_z, steps=50, use_gradient=True, patience=5):
        z = initial_z.copy()
        best_reward = -np.inf
        best_smiles = None
        history = []
        stagnation = 0
        alpha = 0.2
        reward_window = []

        for step in range(steps):
            smiles = self.decoder(z)
            reward = self.reward_fn(smiles)
            self.reward_model.add(z, reward)

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles
                stagnation = 0
            else:
                stagnation += 1

            reward_window.append(reward)
            if len(reward_window) > 5:
                reward_window.pop(0)
                if np.std(reward_window) < 0.01:
                    print("Early stopping due to reward plateau")
                    break

            decoded_smiles = [self.decoder(z + np.random.randn(*z.shape) * 0.01) for _ in range(3)]
            reward += compute_diversity_bonus(decoded_smiles)

            if use_gradient:
                shift = self.reward_model.get_gradient_based_shift(z, alpha)
            else:
                shift = self.reward_model.get_centroid_shift()
                if shift is None:
                    shift = np.random.randn(*z.shape)
                shift = shift - z
                shift = shift / (np.linalg.norm(shift) + 1e-8)

            z += alpha * shift

            if stagnation >= patience:
                alpha *= 0.8  # adaptive alpha
                stagnation = 0

            if step % 5 == 0:
                self.reward_model.train()

            history.append((step, reward, smiles))

        return best_smiles, best_reward, history
