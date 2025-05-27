import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LatentRewardModel:
    """
    LatentRewardModel always runs its neural network model on the CPU to avoid
    accidental GPU memory usage and out-of-memory errors. All tensors are created
    directly on the CPU. After each training session, feedback data is cleared to
    avoid unbounded memory growth. Device placement is explicitly managed for
    debugging and maintainability.
    """
    def __init__(self, latent_dim, hidden_dim=128):
        self.z_list = []
        self.r_list = []
        self.latent_dim = latent_dim
        # Always put model on CPU
        self.device = torch.device("cpu")
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        # print(f"[DEBUG] Model created on device: {next(self.model.parameters()).device}")
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def add(self, z, reward):
        self.z_list.append(np.array(z))
        self.r_list.append(float(reward))

    def train(self, epochs=50):
        if len(self.z_list) < 10:
            print("Not enough data to train.")
            return

        # Always create tensors on CPU
        z_tensor = torch.tensor(np.stack(self.z_list), dtype=torch.float32, device=self.device)
        r_tensor = torch.tensor(self.r_list, dtype=torch.float32, device=self.device).unsqueeze(-1)
        # print(f"[DEBUG] z_tensor device: {z_tensor.device}, r_tensor device: {r_tensor.device}")

        for _ in range(epochs):
            pred = self.model(z_tensor)
            loss = self.loss_fn(pred, r_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Clear data after training to avoid unbounded memory growth
        self.z_list.clear()
        self.r_list.clear()
        # print("[DEBUG] Cleared z_list and r_list after training.")

    def predict(self, z):
        self.model.eval()
        with torch.no_grad():
            # Always create tensor on CPU
            z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
            if len(z_tensor.shape) == 1:
                z_tensor = z_tensor.unsqueeze(0)
            # print(f"[DEBUG] Predict z_tensor device: {z_tensor.device}")
            return self.model(z_tensor).squeeze().cpu().numpy()

    def top_k_z(self, k=50, smiles_list=None):
        """
        Returns up to k latents, ensuring as much uniqueness in SMILES as possible if smiles_list is provided.
        """
        if len(self.z_list) == 0:
            print("[ERROR] No latents in z_list for top_k_z. Returning empty list.")
            return []
        preds = self.predict(np.stack(self.z_list))
        available = len(self.z_list)
        if smiles_list is not None:
            # Prefer unique SMILES in top-k
            seen = set()
            selected = []
            idx_sorted = np.argsort(preds)[::-1]
            for idx in idx_sorted:
                s = smiles_list[idx]
                if s not in seen:
                    selected.append(self.z_list[idx])
                    seen.add(s)
                if len(selected) == k:
                    break
            if len(selected) < k:
                print(f"[WARNING] Only {len(selected)} unique SMILES in top_k_z.")
            return selected
        # fallback to top-k if no SMILES info
        k = min(k, available)
        top_indices = np.argsort(preds)[-k:]
        return [self.z_list[i] for i in top_indices]

    def get_centroid_shift(self, top_k=50, smiles_list=None):
        top_z_list = self.top_k_z(top_k, smiles_list)
        if len(top_z_list) == 0:
            print("[ERROR] No top-z latents found for centroid shift. Returning zero vector.")
            return np.zeros(self.latent_dim)
        top_z = np.stack(top_z_list)
        all_z = np.stack(self.z_list)
        return np.mean(top_z, axis=0) - np.mean(all_z, axis=0)