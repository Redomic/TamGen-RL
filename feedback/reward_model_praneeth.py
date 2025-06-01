# reward_model.py

import numpy as np
import lightgbm as lgb

class LatentRewardModel:
    def __init__(self):
        self.latents = []
        self.rewards = []
        self.model = None
        self.prev_direction = None

    def add(self, z, reward):
        self.latents.append(z)
        self.rewards.append(reward)

    def train(self):
        if len(self.latents) < 10:
            return

        X = np.array(self.latents)
        y = np.array(self.rewards)

        # Normalize rewards for stability
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

        self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
        self.model.fit(X, y)

    def get_gradient_based_shift(self, z, alpha=0.2):
        if self.model is None:
            return np.random.randn(*z.shape) * alpha

        z = z.reshape(1, -1)
        grad = self.model.predict(z, pred_contrib=True)[0][:-1]

        direction = np.array(grad)
        norm = np.linalg.norm(direction)
        direction = direction / (norm + 1e-8)

        # Momentum update
        if self.prev_direction is not None:
            direction = 0.8 * self.prev_direction + 0.2 * direction

        self.prev_direction = direction
        return direction * alpha

    def get_centroid_shift(self):
        if len(self.latents) == 0:
            return None
        return np.mean(self.latents, axis=0)

    def clear_data(self):
        self.latents.clear()
        self.rewards.clear()
