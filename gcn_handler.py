import torch
import numpy as np
from gcn_model_transformer import RelationTransformer 

class GCNHandler:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"[GCN] Loading SOTA Transformer from {weights_path}...")
        
        # Model Mimarisi (Eğitimdeki parametrelerin AYNISI olmalı)
        self.model = RelationTransformer(feature_dim=1024, geo_dim=4).to(device)
        
        # Ağırlıkları Yükle
        checkpoint = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval() # Inference modu (Dropout/Batchnorm kapanır)
        print("[GCN] Model Ready.")

    def predict_score(self, track, detection_feat, detection_bbox):
        """
        Tek bir Track ile Tek bir Detection arasındaki ilişki skorunu (0-1) döndürür.
        """
        # 1. Veri Hazırlığı
        # Track Feature (1024) + Geo (4)
        t_feat = track.last_known_feature
        if t_feat is None:
            return 0.5  # No feature available, return neutral score
        
        # Normalize bbox by image dimensions (assume 1920x1080 for now, or pass as param)
        # Better: normalize to [0,1] range
        t_bbox = np.array(track.last_seen_bbox, dtype=np.float32)
        t_geo = np.array([
            t_bbox[0] / 1920.0,  # x1 normalized
            t_bbox[1] / 1080.0,  # y1 normalized  
            t_bbox[2] / 1920.0,  # x2 normalized
            t_bbox[3] / 1080.0   # y2 normalized
        ], dtype=np.float32)
        
        d_bbox = np.array(detection_bbox, dtype=np.float32)
        d_geo = np.array([
            d_bbox[0] / 1920.0,
            d_bbox[1] / 1080.0,
            d_bbox[2] / 1920.0,
            d_bbox[3] / 1080.0
        ], dtype=np.float32)
        
        # Concat: (1024 + 4) = 1028
        t_input = np.concatenate([t_feat, t_geo])
        d_input = np.concatenate([detection_feat, d_geo])
        
        # 2. Create tensors with CORRECT shape for the model
        # Model expects: (Batch, Dim, N) -> transposes internally to (Batch, N, Dim)
        # So we need: (1, 1028, 1) - Batch=1, Dim=1028, N=1
        
        t_tensor = torch.tensor(t_input, dtype=torch.float32).to(self.device)
        t_tensor = t_tensor.unsqueeze(0).unsqueeze(-1)  # (1028,) -> (1, 1028, 1) ✓

        d_tensor = torch.tensor(d_input, dtype=torch.float32).to(self.device)
        d_tensor = d_tensor.unsqueeze(0).unsqueeze(-1)  # (1028,) -> (1, 1028, 1) ✓
        
        # 3. Inference
        with torch.no_grad():
            logits = self.model(t_tensor, d_tensor)
            score = torch.sigmoid(logits).item()
            
        return score