import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple


class LatentRewardModel:
    """
    Simplified LatentRewardModel for 3-criteria molecular optimization (QED, SAS, Binding Affinity).
    
    NO FALLBACKS - ALL OPERATIONS MUST SUCCEED OR THE MODEL WILL RAISE AN EXCEPTION.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256, device: str = "cpu"):
        self.z_list = []
        self.r_list = []
        self.latent_dim = latent_dim
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Simple neural network architecture
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        print(f"[INFO] LatentRewardModel initialized on device: {self.device}")

    def add(self, z: np.ndarray, reward: float):
        """Add a latent vector and its corresponding reward."""
        if z is None:
            raise ValueError("Latent vector cannot be None")
        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"Invalid reward value: {reward}")
        
        self.z_list.append(np.array(z, dtype=np.float32))
        self.r_list.append(float(reward))

    def train(self, epochs: int = 50) -> dict:
        """
        Train the reward model with simplified training loop.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            dict: Training metrics
            
        Raises:
            ValueError: If insufficient data for training
            RuntimeError: If training fails
        """
        min_samples_required = 20  # Increased minimum for reliable training
        if len(self.z_list) < min_samples_required:
            raise ValueError(f"Insufficient data for training: {len(self.z_list)} samples, need at least {min_samples_required}")

        # Prepare data
        try:
            z_array = np.stack(self.z_list)
            r_array = np.array(self.r_list, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare training data: {e}")
        
        # Validate data
        if np.any(np.isnan(z_array)) or np.any(np.isinf(z_array)):
            raise ValueError("Invalid latent vectors contain NaN or Inf values")
        
        if np.any(np.isnan(r_array)) or np.any(np.isinf(r_array)):
            raise ValueError("Invalid rewards contain NaN or Inf values")
        
        # Normalize rewards for better training
        r_mean, r_std = r_array.mean(), r_array.std()
        if r_std < 1e-6:
            raise ValueError(f"Reward standard deviation too small: {r_std}. Cannot train with constant rewards.")
        
        r_normalized = (r_array - r_mean) / r_std
        
        # Convert to tensors
        try:
            z_tensor = torch.tensor(z_array, dtype=torch.float32, device=self.device)
            r_tensor = torch.tensor(r_normalized, dtype=torch.float32, device=self.device).unsqueeze(-1)
        except Exception as e:
            raise RuntimeError(f"Failed to convert data to tensors: {e}")
        
        # Training loop
        self.model.train()
        initial_loss = None
        
        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()
                pred = self.model(z_tensor)
                loss = self.loss_fn(pred, r_tensor)
                
                if initial_loss is None:
                    initial_loss = loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                # Check for training instability
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"Training became unstable at epoch {epoch}: loss={loss.item()}")
                
            except Exception as e:
                raise RuntimeError(f"Training failed at epoch {epoch}: {e}")
        
        # Store normalization parameters
        self.r_mean = r_mean
        self.r_std = r_std
        
        final_loss = loss.item()
        
        # Validate training success
        if final_loss > initial_loss * 2:  # Loss should not increase significantly
            raise RuntimeError(f"Training failed: loss increased from {initial_loss:.4f} to {final_loss:.4f}")
        
        print(f"[INFO] Training complete. Final loss: {final_loss:.4f}")
        
        return {
            "status": "success",
            "final_loss": final_loss,
            "initial_loss": initial_loss,
            "epochs_trained": epochs,
            "n_samples": len(self.z_list)
        }

    def predict(self, z: np.ndarray) -> np.ndarray:
        """
        Predict reward for given latent vectors.
        
        Args:
            z: Latent vectors to predict rewards for
            
        Returns:
            Predicted rewards (denormalized)
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If prediction fails
        """
        if z is None:
            raise ValueError("Input latent vectors cannot be None")
        
        if not hasattr(self, 'r_mean') or not hasattr(self, 'r_std'):
            raise RuntimeError("Model must be trained before making predictions")
        
        self.model.eval()
        try:
            with torch.no_grad():
                z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
                if len(z_tensor.shape) == 1:
                    z_tensor = z_tensor.unsqueeze(0)
                
                pred_normalized = self.model(z_tensor).squeeze()
                
                # Check for prediction instability
                if torch.any(torch.isnan(pred_normalized)) or torch.any(torch.isinf(pred_normalized)):
                    raise RuntimeError("Model predictions contain NaN or Inf values")
                
                # Denormalize predictions
                pred = pred_normalized * self.r_std + self.r_mean
                
                return pred.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def get_top_k_z(self, k: int = 50) -> List[np.ndarray]:
        """
        Get top-k latent vectors based on predicted rewards.
        
        Args:
            k: Number of top vectors to return
            
        Returns:
            List of top-k latent vectors
            
        Raises:
            ValueError: If no latents are available
            RuntimeError: If prediction fails
        """
        if len(self.z_list) == 0:
            raise ValueError("No latents available for top_k selection")
        
        # Predict rewards for all stored latents
        z_array = np.stack(self.z_list)
        try:
            preds = self.predict(z_array)
        except Exception as e:
            raise RuntimeError(f"Failed to predict rewards for top_k selection: {e}")
        
        # Get top-k indices
        available = len(self.z_list)
        k = min(k, available)
        top_indices = np.argsort(preds)[-k:][::-1]  # Descending order
        
        return [self.z_list[i] for i in top_indices]

    def get_centroid_shift(self, top_k: int = 50) -> np.ndarray:
        """
        Get centroid shift direction for latent space optimization.
        
        Args:
            top_k: Number of top samples to use
            
        Returns:
            Shift direction vector
            
        Raises:
            ValueError: If no latents are available
            RuntimeError: If centroid computation fails
        """
        try:
            top_z_list = self.get_top_k_z(top_k)
        except Exception as e:
            raise RuntimeError(f"Failed to get top-k latents: {e}")
        
        if len(top_z_list) == 0:
            raise ValueError("No top latents available for centroid shift")
        
        try:
            # Simple centroid shift
            top_z = np.stack(top_z_list)
            all_z = np.stack(self.z_list)
            centroid_shift = np.mean(top_z, axis=0) - np.mean(all_z, axis=0)
            
            # Validate shift direction
            if np.any(np.isnan(centroid_shift)) or np.any(np.isinf(centroid_shift)):
                raise RuntimeError("Centroid shift contains NaN or Inf values")
            
            return centroid_shift
        except Exception as e:
            raise RuntimeError(f"Failed to compute centroid shift: {e}")

    def clear_data(self):
        """Clear stored data to free memory."""
        self.z_list.clear()
        self.r_list.clear()
        print("[INFO] Cleared stored latent vectors and rewards.")

    def save_model(self, path: str):
        """Save the trained model."""
        if not hasattr(self, 'r_mean') or not hasattr(self, 'r_std'):
            raise RuntimeError("Model must be trained before saving")
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'r_mean': self.r_mean,
                'r_std': self.r_std,
                'latent_dim': self.latent_dim
            }, path)
            print(f"[INFO] Model saved to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    def load_model(self, path: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.r_mean = checkpoint.get('r_mean', 0.0)
            self.r_std = checkpoint.get('r_std', 1.0)
            print(f"[INFO] Model loaded from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")