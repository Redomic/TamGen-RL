import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LatentRewardModel:
    def __init__(self, latent_dim, hidden_dim=128):
        self.z_list = []
        self.r_list = []
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def add(self, z, reward):
        self.z_list.append(np.array(z))
        self.r_list.append(float(reward))

    def train(self, epochs=50):
        if len(self.z_list) < 10:
            print("Not enough data to train.")
            return

        z_tensor = torch.tensor(np.stack(self.z_list), dtype=torch.float32)
        r_tensor = torch.tensor(self.r_list, dtype=torch.float32).unsqueeze(-1)

        for _ in range(epochs):
            pred = self.model(z_tensor)
            loss = self.loss_fn(pred, r_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, z):
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32)
            if len(z_tensor.shape) == 1:
                z_tensor = z_tensor.unsqueeze(0)
            return self.model(z_tensor).squeeze().numpy()

    def top_k_z(self, k=50):
        preds = self.predict(np.stack(self.z_list))
        top_indices = np.argsort(preds)[-k:]
        return [self.z_list[i] for i in top_indices]

    def get_centroid_shift(self, top_k=50):
        top_z = np.stack(self.top_k_z(top_k))
        all_z = np.stack(self.z_list)
        return np.mean(top_z, axis=0) - np.mean(all_z, axis=0)