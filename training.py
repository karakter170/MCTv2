# train_gating_msmt17_optimized.py
# OPTIMIZED GATING NETWORK TRAINING
#
# Improvements:
# 1. Feature Standardization (Scaling)
# 2. Weighted Loss for Balance
# 3. Enhanced Hard Negative Mining
# 4. OneCycleLR Scheduler
# 5. Dropout tuning

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MSMT17Config:
    """Configuration for MSMT17 training."""

    # Dataset paths - configurable via environment variables or direct setting
    # Set MSMT17_DATASET_ROOT and DINOV3_WEIGHTS_PATH environment variables
    # or update these defaults before training
    dataset_root: str = os.getenv('MSMT17_DATASET_ROOT', './data/MSMT17_V1')

    # DINOv3 pretrained weights path
    pretrained_weights_path: str = os.getenv('DINOV3_WEIGHTS_PATH', './models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

    feature_extractor: str = "dinov3_vitl16"
    feature_dim: int = 1024
    batch_size_extraction: int = 64 # Increased for speed
    
    # Training data generation - INCREASED DATASET SIZE
    samples_per_identity: int = 100 # More samples per ID
    hard_negative_ratio: float = 0.8 # Focus heavily on hard examples
    
    # Simulated track parameters
    min_track_length: int = 5
    max_track_length: int = 50
    quality_noise_std: float = 0.15
    
    # Training Hyperparameters
    batch_size: int = 256 # Larger batch size for stability
    epochs: int = 50
    learning_rate: float = 3e-4 # Slightly lower starting LR
    weight_decay: float = 1e-4
    
    # Output
    output_dir: str = "gating_training_optimized"
    model_save_path: str = "models/gating_network_msmt17.pt"


# =============================================================================
# MSMT17 DATASET LOADER (Same as before)
# =============================================================================

class MSMT17Dataset:
    def __init__(self, root_path: str, split: str = "train"):
        self.root = Path(root_path)
        self.split = split
        self.split_path = self.root / split
        self.data = self._load_data()
        self.identities = list(self.data.keys())
        print(f"[MSMT17] Loaded {split}: {len(self.identities)} identities, {sum(len(v) for v in self.data.values())} images")
    
    def _load_data(self) -> Dict[int, List[dict]]:
        data = defaultdict(list)
        if not self.split_path.exists(): raise FileNotFoundError(f"Split path not found: {self.split_path}")
        for identity_folder in sorted(self.split_path.iterdir()):
            if not identity_folder.is_dir(): continue
            try: identity_id = int(identity_folder.name)
            except ValueError: continue
            for img_path in sorted(identity_folder.glob("*.jpg")):
                data[identity_id].append({'path': str(img_path), 'identity': identity_id})
        return dict(data)
    
    def get_images_for_identity(self, identity_id: int) -> List[dict]:
        return self.data.get(identity_id, [])
    
    def get_identities_with_min_images(self, min_images: int) -> List[int]:
        return [pid for pid, images in self.data.items() if len(images) >= min_images]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    def __init__(self, config: MSMT17Config, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config.feature_extractor
        self.batch_size = config.batch_size_extraction  # Config'den batch size al
        
        print(f"[FeatureExtractor] Loading {self.model_name} on {self.device}...")
        
        try:
            self.model = torch.hub.load('facebookresearch/dinov3', self.model_name, pretrained=False)
        except:
            print("[FeatureExtractor] Switching to DINOv2 fallback...")
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, pretrained=False)

        if os.path.exists(config.pretrained_weights_path):
            # Weights_only uyarısını susturmak için True ekledik
            state_dict = torch.load(config.pretrained_weights_path, map_location='cpu', weights_only=True)
            
            if 'model' in state_dict: state_dict = state_dict['model']
            elif 'teacher' in state_dict: state_dict = state_dict['teacher']
            
            self.model.load_state_dict(state_dict, strict=False)
            print("[FeatureExtractor] Local weights loaded.")
        else:
            raise FileNotFoundError(f"Weights not found: {config.pretrained_weights_path}")
        
        self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for img in images:
            if isinstance(img, np.ndarray): img = Image.fromarray(img)
            img = img.resize((224, 224), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            processed.append(img_tensor)
        
        if not processed:
            return None
            
        batch = torch.stack(processed).to(self.device)
        return (batch - self.mean) / self.std
    
    @torch.no_grad()
    def extract_from_paths(self, paths: List[str]) -> np.ndarray:
        all_features = []
        # BURAYA TQDM (İLERLEME ÇUBUĞU) EKLENDİ
        print(f"[FeatureExtractor] Extracting features from {len(paths)} images...")
        
        for i in tqdm(range(0, len(paths), self.batch_size), desc="Extracting Batches"):
            batch_paths = paths[i:i + self.batch_size]
            images = []
            valid_paths = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    images.append(np.array(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Error reading image {p}: {e}")

            if not images:
                continue

            batch = self.preprocess(images)
            if batch is not None:
                # Mixed Precision (autocast) for speedup and memory savings
                # Only use CUDA autocast if CUDA is available
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        features = self.model(batch)
                        features = F.normalize(features, p=2, dim=1)
                else:
                    features = self.model(batch)
                    features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu().numpy())
                
        if not all_features:
            return np.array([])
            
        return np.vstack(all_features)


# =============================================================================
# SIMULATION LOGIC (Same logic, cleaner implementation)
# =============================================================================

@dataclass
class SimulatedTrackState:
    slow_mean: np.ndarray
    slow_variance: np.ndarray
    fast_buffer: List[np.ndarray]
    count: int
    consistency_history: List[float]
    divergence_counter: int
    track_age: float
    time_since_last_update: float

def simulate_track(features: List[np.ndarray]) -> SimulatedTrackState:
    if not features: return None
    slow_mean = features[0].copy()
    slow_variance = np.ones_like(slow_mean) * 0.05
    fast_buffer = [features[0].copy()]
    consistency_history = []
    div_counter = 0
    alpha = 0.05
    
    for i in range(1, len(features)):
        feat = features[i]
        sim = float(np.dot(feat, slow_mean))
        consistency_history.append(sim)
        fast_buffer.append(feat)
        if len(fast_buffer) > 7: fast_buffer.pop(0)
        
        if sim > 0.65:
            slow_mean = (1-alpha)*slow_mean + alpha*feat
            slow_mean /= (np.linalg.norm(slow_mean) + 1e-8)
            div_counter = max(0, div_counter - 1)
        else:
            div_counter += 1
            
    return SimulatedTrackState(
        slow_mean, slow_variance, fast_buffer, len(features),
        consistency_history, div_counter, len(features)*1.0, random.uniform(0.1, 5.0)
    )

def compute_context(query: np.ndarray, quality: float, state: SimulatedTrackState) -> np.ndarray:
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    s_norm = state.slow_mean / (np.linalg.norm(state.slow_mean) + 1e-8)
    
    cosine = float(np.dot(q_norm, s_norm))
    l2 = float(np.linalg.norm(q_norm - s_norm)) / 2.0
    
    if state.fast_buffer:
        buf_sim = np.mean([np.dot(q_norm, b/(np.linalg.norm(b)+1e-8)) for b in state.fast_buffer])
    else:
        buf_sim = cosine
        
    cons_ema = 0.5
    if state.consistency_history:
        cons_ema = np.mean(state.consistency_history[-10:])
        
    # --- CRITICAL: Raw values here, Scaling happens in Trainer ---
    return np.array([
        cosine,                 # 0: Cosine Sim (-1 to 1)
        l2,                     # 1: L2 Dist (0 to 1)
        quality,                # 2: Quality (0 to 1)
        float(buf_sim),         # 3: Buffer Sim (-1 to 1)
        state.track_age / 300.0, # 4: Age (Normalized approx)
        state.time_since_last_update / 60.0, # 5: Time since (Normalized)
        state.count / 100.0,    # 6: Count (Normalized)
        min(1.0, state.count/100.0), # 7: Maturity
        0.05,                   # 8: Variance (Placeholder)
        cons_ema,               # 9: Consistency
        state.divergence_counter / 30.0, # 10: Divergence Ratio
        quality                 # 11: Quality History (Approx)
    ], dtype=np.float32)


# =============================================================================
# DATA GENERATOR WITH IMPROVED MINING
# =============================================================================

class DataGenerator:
    def __init__(self, dataset, extractor, config):
        self.dataset = dataset
        self.extractor = extractor
        self.config = config
        self.feats = {} # path -> feat
        self.id_feats = defaultdict(list)
        
    def precompute(self, cache_path):
        if os.path.exists(cache_path):
            print("Loading features from cache...")
            with open(cache_path, 'rb') as f: data = pickle.load(f)
            self.feats = data['feats']
            self.id_feats = data['id_feats']
            return

        print("Extracting features...")
        paths = []
        path_to_pid = {}  # FIXED: O(1) lookup instead of O(N*M)

        # Build path list and mapping in single pass
        for pid, imgs in self.dataset.data.items():
            for img in imgs:
                path = img['path']
                paths.append(path)
                path_to_pid[path] = pid

        feats_arr = self.extractor.extract_from_paths(paths)

        # Use efficient dict lookup instead of nested loops
        for i, path in enumerate(paths):
            self.feats[path] = feats_arr[i]
            pid = path_to_pid[path]  # O(1) lookup
            self.id_feats[pid].append(feats_arr[i])

        with open(cache_path, 'wb') as f:
            pickle.dump({'feats': self.feats, 'id_feats': self.id_feats}, f)

    def generate(self, n_samples):
        contexts, labels = [], []
        pids = [p for p, fs in self.id_feats.items() if len(fs) > 5]
        
        print("Generating Positive Samples...")
        for _ in tqdm(range(n_samples // 2)):
            pid = random.choice(pids)
            feats = self.id_feats[pid]
            if len(feats) < 5: continue
            
            # Split: History vs Query
            idx = random.randint(4, len(feats)-1)
            hist = feats[:idx]
            query = feats[idx]
            
            state = simulate_track(hist)
            ctx = compute_context(query, random.uniform(0.7, 1.0), state)
            contexts.append(ctx)
            labels.append(1.0)

        print("Generating Negative Samples (Hard Mining)...")
        n_neg = n_samples - len(contexts)
        for _ in tqdm(range(n_neg)):
            pid = random.choice(pids)
            feats = self.id_feats[pid]
            hist = feats[:random.randint(4, min(len(feats), 20))]
            state = simulate_track(hist)
            
            # Hard Negative Mining
            if random.random() < self.config.hard_negative_ratio:
                # Find a confusing ID (high dot product with mean)
                best_wrong_pid = None
                best_wrong_sim = -1.0
                
                # Try 10 random other IDs
                for _ in range(10):
                    wrong_pid = random.choice(pids)
                    if wrong_pid == pid: continue
                    wrong_feat = random.choice(self.id_feats[wrong_pid])
                    sim = np.dot(state.slow_mean, wrong_feat)
                    if sim > best_wrong_sim:
                        best_wrong_sim = sim
                        best_wrong_pid = wrong_pid
                        best_wrong_feat = wrong_feat
                
                query = best_wrong_feat
            else:
                # Easy negative
                wrong_pid = random.choice(pids)
                while wrong_pid == pid: wrong_pid = random.choice(pids)
                query = random.choice(self.id_feats[wrong_pid])
            
            ctx = compute_context(query, random.uniform(0.7, 1.0), state)
            contexts.append(ctx)
            labels.append(0.0)
            
        return np.array(contexts), np.array(labels)


# =============================================================================
# OPTIMIZED GATING NETWORK & TRAINER
# =============================================================================

class GatingNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=None):
        super().__init__()
        # CRITICAL: Must match learned_gating.py architecture exactly
        # for trained weights to be compatible with inference
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm (not BatchNorm!)
                nn.ReLU(),
                nn.Dropout(0.1)  # Consistent 0.1 dropout
            ])
            prev_dim = hidden_dim

        # Output layer with Sigmoid (part of the model architecture)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = GatingNetwork().to(device)
        self.scaler = StandardScaler() # Crucial for convergence
        
    def fit(self, X, y):
        # 1. Scale Features
        print("[Trainer] Fitting scaler to data...")
        X = self.scaler.fit_transform(X)
        
        # Save scaler for inference later
        with open(os.path.join(self.config.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # 2. Convert to Tensor
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )
        
        # Split
        train_len = int(0.85 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])
        
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size)
        
        # 3. Setup Optimization
        # Use BCELoss since Sigmoid is built into the model architecture
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=self.config.epochs
        )
        
        print(f"[Trainer] Starting training on {self.device}...")
        best_acc = 0.0
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss, correct = 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    probs = self.model(bx)  # Already includes sigmoid
                    loss = criterion(probs, by)
                    val_loss += loss.item()

                    preds = (probs > 0.5).float()  # No sigmoid needed
                    correct += (preds == by).sum().item()
            
            acc = correct / len(val_set)
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} | "
                      f"Loss: {train_loss/len(train_loader):.4f} | "
                      f"Val Loss: {val_loss/len(val_loader):.4f} | "
                      f"Val Acc: {acc*100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                # Save model state dict directly (architecture already includes sigmoid)
                self.save_inference_model()

    def save_inference_model(self):
        # Save the model's state_dict directly
        # The model architecture already includes Sigmoid, so no wrapper needed
        # This format is compatible with learned_gating.py loading
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.config.model_save_path)
        print(f"[Trainer] Model saved to {self.config.model_save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = MSMT17Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Setup
    dataset = MSMT17Dataset(config.dataset_root)
    extractor = FeatureExtractor(config)
    generator = DataGenerator(dataset, extractor, config)
    
    # 2. Cache Features
    generator.precompute(os.path.join(config.output_dir, "features.pkl"))
    
    # 3. Generate Data
    X, y = generator.generate(n_samples=50000) # Ensure enough data
    print(f"Data shape: {X.shape}, Balance: {np.mean(y):.2f}")
    
    # 4. Train
    trainer = Trainer(config)
    trainer.fit(X, y)
    print(f"\n[Done] Model saved to {config.model_save_path}")

if __name__ == "__main__":
    main()