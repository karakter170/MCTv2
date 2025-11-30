import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gcn_model_sota import CrossGCN

# --- SETTINGS ---
DATA_DIR = "./processed_mot17"
MODEL_SAVE_PATH = "models/sota_gcn_dinov3_5dim.pth"
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001  # Bias Init olduğu için güvenli, çok düşük yapmaya gerek yok.

# --- FOCAL LOSS CLASS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.90, gamma=2.0):
        super(FocalLoss, self).__init__()
        # Pozitifleri (1) bulmak daha önemli olduğu için alpha yüksek (0.90)
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, mask=None):
        # inputs: Logits (Modelden gelen ham puan)
        # targets: 0 veya 1
        bce_loss = self.bce(inputs, targets)
        probas = torch.sigmoid(inputs) # Logit -> Prob
        
        p_t = probas * targets + (1 - probas) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()

class MOT17SmartDataset(Dataset):
    def __init__(self, data_dir, time_gap_max=30, mode='train', train_split=0.8):
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        all_files.sort()  # Ensure consistent ordering

        # Split by sequences (not pairs) to avoid data leakage
        split_idx = int(len(all_files) * train_split)
        if mode == 'train':
            self.files = all_files[:split_idx]
        elif mode == 'val':
            self.files = all_files[split_idx:]
        else:
            self.files = all_files

        self.samples = []
        self.time_gap_max = time_gap_max
        print(f"[{mode.upper()}] Akıllı veri çiftleri oluşturuluyor (Smart Sampling)...")
        print(f"[{mode.upper()}] Using {len(self.files)} sequences")
        self._prepare_data()

    def _prepare_data(self):
        valid_pairs = 0
        max_det_a = 0
        max_det_b = 0
        total_positives = 0
        total_elements = 0

        for f_path in self.files:
            try:
                seq_data = np.load(f_path, allow_pickle=True)
            except: continue

            for i in range(len(seq_data) - 1):
                frame_a = seq_data[i]
                ids_a = set([m['id'] for m in frame_a['meta']])
                if len(ids_a) == 0: continue

                search_indices = list(range(1, min(self.time_gap_max, len(seq_data) - i)))

                pair_count_for_frame = 0 # Sayaç

                for gap in search_indices:
                    frame_b = seq_data[i + gap]
                    ids_b = set([m['id'] for m in frame_b['meta']])
                    if len(ids_b) == 0: continue

                    common_ids = ids_a.intersection(ids_b)

                    if len(common_ids) >= 1:
                        self.samples.append((frame_a, frame_b))
                        valid_pairs += 1
                        pair_count_for_frame += 1

                        # Track statistics
                        max_det_a = max(max_det_a, len(ids_a))
                        max_det_b = max(max_det_b, len(ids_b))
                        total_positives += len(common_ids)
                        total_elements += len(ids_a) * len(ids_b)

                        # --- CRITICAL FIX ---
                        # Bir frame için 3 örnek bulduysan dur, diğer frame'e geç.
                        # Veri setini gereksiz şişirmeyi engeller.
                        if pair_count_for_frame >= 3:
                            break

        pos_ratio = total_positives / (total_elements + 1e-6)
        print(f"Toplam {valid_pairs} adet 'Dolu' eğitim çifti bulundu.")
        print(f"Max detections: Frame A={max_det_a}, Frame B={max_det_b}")
        print(f"Positive ratio: {pos_ratio:.4f} (class imbalance: {1/pos_ratio:.1f}:1)")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fa, fb = self.samples[idx]

        # 1. Delta Time
        delta_frames = abs(fb['frame'] - fa['frame'])
        delta_seconds = delta_frames / 30.0
        norm_dt = np.tanh(delta_seconds / 10.0)

        # 2. Features & Geo
        feat_a = fa['features']; feat_b = fb['features']

        # Geo verisi prepare_mot17.py ile [0-1] arası geliyor.
        # FIX: Scale to [-1, 1] to match DINOv3 feature range
        geo_a = np.array([m['geo'] for m in fa['meta']], dtype=np.float32)
        geo_b = np.array([m['geo'] for m in fb['meta']], dtype=np.float32)
        geo_a = (geo_a - 0.5) * 2.0  # [0,1] -> [-1,1]
        geo_b = (geo_b - 0.5) * 2.0  # [0,1] -> [-1,1]

        # 3. Concatenate Time (5. Boyut) - SYMMETRIC encoding
        # Frame A gets negative time, Frame B gets positive (relative encoding)
        dt_col_a = np.full((geo_a.shape[0], 1), -norm_dt/2, dtype=np.float32)
        geo_a = np.concatenate([geo_a, dt_col_a], axis=1)

        dt_col_b = np.full((geo_b.shape[0], 1), +norm_dt/2, dtype=np.float32)
        geo_b = np.concatenate([geo_b, dt_col_b], axis=1)

        # 4. Input Concat (1024 + 5 = 1029)
        input_a = np.concatenate([feat_a, geo_a], axis=1)
        input_b = np.concatenate([feat_b, geo_b], axis=1)
        
        # 5. Labels
        ids_a = [m['id'] for m in fa['meta']]
        ids_b = [m['id'] for m in fb['meta']]
        
        labels = np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
        for r, ida in enumerate(ids_a):
            for c, idb in enumerate(ids_b):
                if ida == idb:
                    labels[r, c] = 1.0
        
        # 6. Padding & Masking
        MAX_N = 64
        input_a = self._pad(input_a, MAX_N)
        input_b = self._pad(input_b, MAX_N)
        labels = self._pad_matrix(labels, MAX_N)
        
        mask = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        real_r = min(len(ids_a), MAX_N)
        real_c = min(len(ids_b), MAX_N)
        mask[:real_r, :real_c] = 1.0
        
        return torch.tensor(input_a).T, torch.tensor(input_b).T, torch.tensor(labels), torch.tensor(mask)

    def _pad(self, arr, max_n):
        curr = arr.shape[0]
        if curr >= max_n: return arr[:max_n, :]
        padded = np.zeros((max_n, arr.shape[1]), dtype=np.float32)
        padded[:curr, :] = arr
        return padded

    def _pad_matrix(self, mat, max_n):
        r, c = mat.shape
        r, c = min(r, max_n), min(c, max_n)
        padded = np.zeros((max_n, max_n), dtype=np.float32)
        padded[:r, :c] = mat[:r, :c]
        return padded

def validate(model, val_loader, criterion, device):
    """Run validation and return metrics"""
    model.eval()
    total_loss = 0
    total_pos_avg = 0
    total_neg_avg = 0
    valid_batches = 0

    with torch.no_grad():
        for trks, dets, lbls, masks in val_loader:
            trks, dets, lbls, masks = trks.to(device), dets.to(device), lbls.to(device), masks.to(device)

            logits = model(trks, dets)
            loss = criterion(logits, lbls, masks)

            total_loss += loss.item()
            valid_batches += 1

            # Statistics
            probs = torch.sigmoid(logits)
            true_pos = probs[lbls == 1]
            true_neg = probs[lbls == 0]

            if len(true_pos) > 0:
                total_pos_avg += true_pos.mean().item()
            if len(true_neg) > 0:
                total_neg_avg += true_neg.mean().item()

    avg_loss = total_loss / (valid_batches + 1e-6)
    avg_pos = total_pos_avg / (valid_batches + 1e-6)
    avg_neg = total_neg_avg / (valid_batches + 1e-6)

    return avg_loss, avg_pos, avg_neg

def train():
    os.makedirs("models", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Datasets - Train/Val split
    train_dataset = MOT17SmartDataset(DATA_DIR, mode='train', train_split=0.8)
    val_dataset = MOT17SmartDataset(DATA_DIR, mode='val', train_split=0.8)

    # DataLoaders - Increased num_workers, removed drop_last
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)
    
    # Model (1029 Dim)
    model = CrossGCN(feature_dim=1029).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # Learning Rate Scheduler - Reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=5, verbose=True, min_lr=1e-6)

    # Focal Loss (Alpha 0.90 çünkü Pozitif örnekler çok değerli)
    criterion = FocalLoss(alpha=0.90, gamma=2.0).to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    print(f"--- EĞİTİM BAŞLIYOR (LR={LR}, Input=1029, Output=Logits) ---")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(EPOCHS):
        # ===== TRAINING PHASE =====
        model.train()
        total_loss = 0
        valid_batches = 0

        for batch_i, (trks, dets, lbls, masks) in enumerate(train_loader):
            trks, dets, lbls, masks = trks.to(device), dets.to(device), lbls.to(device), masks.to(device)

            optimizer.zero_grad()

            # 1. Modelden HAM LOGIT al (Sigmoid yok!)
            logits = model(trks, dets)

            # 2. Loss hesapla (Logits -> Focal Loss)
            loss = criterion(logits, lbls, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            # 3. İstatistik (SADECE burası için Sigmoid al)
            probs = torch.sigmoid(logits)

            true_pos = probs[lbls == 1]
            true_neg = probs[lbls == 0]

            pos_avg = true_pos.mean().item() if len(true_pos) > 0 else 0.0
            neg_avg = true_neg.mean().item() if len(true_neg) > 0 else 0.0

            if batch_i % 50 == 0:
                print(f"Ep {epoch+1} | Batch {batch_i} | Loss: {loss.item():.4f} | Pos_Avg: {pos_avg:.4f} | Neg_Avg: {neg_avg:.4f}")

        train_loss = total_loss / (valid_batches + 1e-6)

        # ===== VALIDATION PHASE =====
        val_loss, val_pos, val_neg = validate(model, val_loader, criterion, device)

        print(f">>> Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pos: {val_pos:.4f} | Val Neg: {val_neg:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered! Best Val Loss: {best_val_loss:.4f}")
            break

    print("Eğitim Tamamlandı.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()