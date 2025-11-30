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
    def __init__(self, data_dir, time_gap_max=30):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.samples = []
        self.time_gap_max = time_gap_max
        print("Akıllı veri çiftleri oluşturuluyor (Smart Sampling)...")
        self._prepare_data()

    def _prepare_data(self):
        valid_pairs = 0
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
                        
                        # --- CRITICAL FIX ---
                        # Bir frame için 3 örnek bulduysan dur, diğer frame'e geç.
                        # Veri setini gereksiz şişirmeyi engeller.
                        if pair_count_for_frame >= 3:
                            break 
                        
        print(f"Toplam {valid_pairs} adet 'Dolu' eğitim çifti bulundu.")
    
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
        
        # Geo verisi prepare_mot17.py ile zaten normalize (0-1 arası) geliyor.
        # EKSTRA İŞLEM YAPMA!
        geo_a = np.array([m['geo'] for m in fa['meta']], dtype=np.float32)
        geo_b = np.array([m['geo'] for m in fb['meta']], dtype=np.float32)
        
        # 3. Concatenate Time (5. Boyut)
        dt_col_a = np.full((geo_a.shape[0], 1), norm_dt, dtype=np.float32)
        geo_a = np.concatenate([geo_a, dt_col_a], axis=1)
        
        dt_col_b = np.full((geo_b.shape[0], 1), 0.0, dtype=np.float32)
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

def train():
    os.makedirs("models", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset
    dataset = MOT17SmartDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    
    # Model (1029 Dim)
    model = CrossGCN(feature_dim=1029).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Focal Loss (Alpha 0.90 çünkü Pozitif örnekler çok değerli)
    criterion = FocalLoss(alpha=0.90, gamma=2.0).to(device)
    
    print(f"--- EĞİTİM BAŞLIYOR (LR={LR}, Input=1029, Output=Logits) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_i, (trks, dets, lbls, masks) in enumerate(dataloader):
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
            neg_avg = true_neg.mean().item()
            
            if batch_i % 50 == 0:
                print(f"Ep {epoch+1} | Batch {batch_i} | Loss: {loss.item():.4f} | Pos_Avg: {pos_avg:.4f} | Neg_Avg: {neg_avg:.4f}")
        
        avg_loss = total_loss / (valid_batches + 1e-6)
        print(f">>> Epoch {epoch+1} Bitti. Avg Loss: {avg_loss:.4f}")
        
        # Loss 0.03 altına inerse kaydet
        if (epoch+1) % 5 == 0 or avg_loss < 0.03:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model Kaydedildi: {MODEL_SAVE_PATH}")

    print("Eğitim Tamamlandı.")

if __name__ == "__main__":
    train()