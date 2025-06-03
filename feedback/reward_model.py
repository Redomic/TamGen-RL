import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple


class LatentRewardModel:
    """
    Enhanced LatentRewardModel for molecular property prediction in latent space.
    Fixed data flow issues and improved architecture for better performance.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256, device: str = "cpu"):
        self.z_list = []
        self.r_list = []
        self.latent_dim = latent_dim
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Improved neural network architecture
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        ).to(self.device)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(hidden_dim // 4, 1).to(self.device)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        print(f"[INFO] LatentRewardModel initialized on device: {self.device}")

    def add(self, z: np.ndarray, reward: float):
        """Add a latent vector and its corresponding reward."""
        self.z_list.append(np.array(z, dtype=np.float32))
        self.r_list.append(float(reward))

    def train(self, epochs: int = 100, validation_split: float = 0.2) -> dict:
        """
        Train the reward model with validation and early stopping.
        
        Returns:
            dict: Training metrics
        """
        if len(self.z_list) < 20:
            print(f"[WARNING] Only {len(self.z_list)} samples available for training. Consider gathering more data.")
            return {"status": "insufficient_data"}

        # Prepare data
        z_array = np.stack(self.z_list)
        r_array = np.array(self.r_list, dtype=np.float32)
        
        # Normalize rewards for better training
        r_mean, r_std = r_array.mean(), r_array.std()
        if r_std > 1e-6:
            r_normalized = (r_array - r_mean) / r_std
        else:
            r_normalized = r_array - r_mean
        
        # Train-validation split
        n_samples = len(z_array)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:-n_val] if n_val > 0 else indices
        val_indices = indices[-n_val:] if n_val > 0 else []
        
        # Convert to tensors
        z_train = torch.tensor(z_array[train_indices], dtype=torch.float32, device=self.device)
        r_train = torch.tensor(r_normalized[train_indices], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        if len(val_indices) > 0:
            z_val = torch.tensor(z_array[val_indices], dtype=torch.float32, device=self.device)
            r_val = torch.tensor(r_normalized[val_indices], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        train_losses = []
        val_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            # Training step
            self.optimizer.zero_grad()
            pred = self.model(z_train)
            loss = self.loss_fn(pred, r_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation step
            if len(val_indices) > 0 and epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(z_val)
                    val_loss = self.loss_fn(val_pred, r_val)
                    val_losses.append(val_loss.item())
                    
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"[INFO] Early stopping at epoch {epoch}")
                        break
                        
                self.model.train()
        
        # Store normalization parameters for prediction
        self.r_mean = r_mean
        self.r_std = r_std if r_std > 1e-6 else 1.0
        
        print(f"[INFO] Training complete. Final train loss: {train_losses[-1]:.4f}")
        
        return {
            "status": "success",
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None,
            "epochs_trained": epoch + 1,
            "n_samples": len(self.z_list)
        }

    def predict(self, z: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """
        Predict reward for given latent vectors.
        
        Args:
            z: Latent vectors to predict rewards for
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predicted rewards (denormalized)
        """
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
            if len(z_tensor.shape) == 1:
                z_tensor = z_tensor.unsqueeze(0)
            
            pred_normalized = self.model(z_tensor).squeeze()
            
            # Denormalize predictions
            if hasattr(self, 'r_mean') and hasattr(self, 'r_std'):
                pred = pred_normalized * self.r_std + self.r_mean
            else:
                pred = pred_normalized
            
            result = pred.cpu().numpy()
            
            if return_uncertainty:
                # Simple uncertainty estimation using dropout
                self.model.train()
                uncertainties = []
                for _ in range(10):  # Monte Carlo dropout
                    with torch.no_grad():
                        mc_pred = self.model(z_tensor).squeeze()
                        if hasattr(self, 'r_mean') and hasattr(self, 'r_std'):
                            mc_pred = mc_pred * self.r_std + self.r_mean
                        uncertainties.append(mc_pred.cpu().numpy())
                
                uncertainty = np.std(uncertainties, axis=0)
                self.model.eval()
                return result, uncertainty
            
            return result

    def get_top_k_z_before_clear(self, k: int = 50, smiles_list: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Get top-k latent vectors BEFORE clearing data.
        This fixes the critical data flow bug.
        """
        if len(self.z_list) == 0:
            print("[ERROR] No latents available for top_k selection.")
            return []
        
        # Predict rewards for all stored latents
        z_array = np.stack(self.z_list)
        preds = self.predict(z_array)
        
        available = len(self.z_list)
        k = min(k, available)
        
        if smiles_list is not None and len(smiles_list) == len(self.z_list):
            # Prefer unique SMILES in top-k
            seen_smiles = set()
            selected = []
            idx_sorted = np.argsort(preds)[::-1]  # Descending order
            
            for idx in idx_sorted:
                smi = smiles_list[idx]
                if smi not in seen_smiles:
                    selected.append(self.z_list[idx])
                    seen_smiles.add(smi)
                if len(selected) >= k:
                    break
            
            if len(selected) < k:
                print(f"[WARNING] Only {len(selected)} unique SMILES found for top-k selection.")
            return selected
        else:
            # Fallback to simple top-k
            top_indices = np.argsort(preds)[-k:][::-1]  # Descending order
            return [self.z_list[i] for i in top_indices]

    def get_gradient_based_shift(self, top_k: int = 50, lr: float = 0.01, steps: int = 100) -> np.ndarray:
        """
        Compute optimal shift direction using gradient-based optimization in latent space.
        
        Args:
            top_k: Number of top samples to use as starting point
            lr: Learning rate for optimization
            steps: Number of optimization steps
            
        Returns:
            Optimal shift direction vector
        """
        if len(self.z_list) == 0:
            print("[ERROR] No latents available for gradient-based shift.")
            return np.zeros(self.latent_dim)
        
        # Get top-k starting points
        top_z_list = self.get_top_k_z_before_clear(top_k)
        if len(top_z_list) == 0:
            return np.zeros(self.latent_dim)
        
        # Convert to tensor and enable gradients
        z_tensor = torch.tensor(np.stack(top_z_list), dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z_tensor], lr=lr)
        
        self.model.eval()
        initial_mean = z_tensor.detach().clone().mean(dim=0)
        
        # Optimize in latent space to maximize predicted reward
        for step in range(steps):
            optimizer.zero_grad()
            pred_rewards = self.model(z_tensor).squeeze()
            
            # Maximize reward while adding regularization to prevent too large shifts
            reward_loss = -pred_rewards.mean()
            reg_loss = 0.01 * torch.norm(z_tensor - initial_mean, dim=1).mean()
            
            total_loss = reward_loss + reg_loss
            total_loss.backward()
            optimizer.step()
        
        # Compute final shift direction
        optimized_mean = z_tensor.detach().mean(dim=0)
        shift_direction = (optimized_mean - initial_mean).cpu().numpy()
        
        return shift_direction

    def get_centroid_shift(self, top_k: int = 50, smiles_list: Optional[List[str]] = None, 
                          use_gradient: bool = True) -> np.ndarray:
        """
        Get centroid shift direction for latent space optimization.
        
        Args:
            top_k: Number of top samples to use
            smiles_list: Optional SMILES list for diversity consideration
            use_gradient: Whether to use gradient-based optimization
            
        Returns:
            Shift direction vector
        """
        if use_gradient:
            return self.get_gradient_based_shift(top_k)
        else:
            # Simple centroid shift (original method)
            top_z_list = self.get_top_k_z_before_clear(top_k, smiles_list)
            if len(top_z_list) == 0:
                return np.zeros(self.latent_dim)
            
            top_z = np.stack(top_z_list)
            all_z = np.stack(self.z_list)
            return np.mean(top_z, axis=0) - np.mean(all_z, axis=0)

    def clear_data(self):
        """Clear stored data to free memory."""
        self.z_list.clear()
        self.r_list.clear()
        print("[INFO] Cleared stored latent vectors and rewards.")

    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'r_mean': getattr(self, 'r_mean', 0.0),
            'r_std': getattr(self, 'r_std', 1.0),
            'latent_dim': self.latent_dim
        }, path)
        print(f"[INFO] Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.r_mean = checkpoint.get('r_mean', 0.0)
        self.r_std = checkpoint.get('r_std', 1.0)
        print(f"[INFO] Model loaded from {path}")