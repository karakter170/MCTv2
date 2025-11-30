import torch
import numpy as np
from gcn_model_transformer import RelationTransformer 

class RelationRefiner:  # Renamed from GCNHandler for accuracy
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"[RelationRefiner] Loading SOTA Transformer from {weights_path}...")
        
        # Model Architecture (Must match training)
        self.model = RelationTransformer(feature_dim=1024, geo_dim=4).to(device)
        
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"[RelationRefiner] Warning: Could not load weights: {e}")
            
        self.model.eval()
        print("[RelationRefiner] Model Ready (Batch Mode Enabled).")

    def _normalize_bbox(self, bbox, w, h):
        """Normalize bbox to [0, 1] range."""
        return np.array([
            bbox[0] / w, bbox[1] / h,
            bbox[2] / w, bbox[3] / h
        ], dtype=np.float32)

    def predict_batch(self, track, candidates, frame_w, frame_h, curr_time):
        """
        Batch Inference: Compare 1 Track against N Candidates.
        
        Args:
            track: The GlobalTrack object (Query)
            candidates: List of dicts/objects with 'feature' and 'bbox' (Keys)
            frame_w, frame_h: Dimensions of the current camera frame
        
        Returns:
            scores: Numpy array of shape (N,) with similarity probabilities.
        """
        if not candidates:
            return np.array([])
        
        dt = curr_time - track.last_seen_timestamp

        norm_dt = np.tanh(dt / 10.0)

        t_geo = np.concatenate([
            self._normalize_bbox(track.last_seen_bbox, 1920, 1080), 
            [norm_dt]
        ])

        # 1. Prepare Track Data (Query)
        # Use robust ID (Slow Memory) if available, else Fast Memory
        t_feat = track.robust_id if track.robust_id is not None else track.last_known_feature
        if t_feat is None: return np.zeros(len(candidates))
        
        t_bbox = track.last_seen_bbox
        # Note: Track bbox is from a previous camera/time. 
        # Ideally, we should normalize it by *that* camera's resolution.
        # For now, we assume standard normalization or similar aspect ratios.
        t_geo = self._normalize_bbox(t_bbox, 1920, 1080) # Fallback if prev cam res unknown
        
        t_input = np.concatenate([t_feat, t_geo])
        
        # 2. Prepare Candidates Data (Keys)
        d_inputs = []
        for cand in candidates:
            d_feat = cand['feature']
            d_geo = self._normalize_bbox(cand['bbox'], frame_w, frame_h)
            d_inputs.append(np.concatenate([d_feat, d_geo]))
            
        # 3. Batch Tensor Creation
        # Shape: (1, 1028, 1) -> Batch=1, Dim=1028, N=1 Track
        t_tensor = torch.tensor(t_input, dtype=torch.float32).to(self.device)
        t_tensor = t_tensor.unsqueeze(0).unsqueeze(-1) 

        # Shape: (1, 1028, M) -> Batch=1, Dim=1028, M Candidates
        d_stack = np.stack(d_inputs, axis=1) # (1028, M)
        d_tensor = torch.tensor(d_stack, dtype=torch.float32).to(self.device)
        d_tensor = d_tensor.unsqueeze(0) 

        # 4. Inference
        with torch.no_grad():
            # The model handles broadcasting internally via .expand()
            logits = self.model(t_tensor, d_tensor) # Output: (1, 1, M)
            scores = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        # Handle single candidate case returning scalar
        if scores.ndim == 0:
            scores = np.array([scores])
            
        return scores