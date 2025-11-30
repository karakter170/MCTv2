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
            curr_time: Current timestamp for temporal features

        Returns:
            scores: Numpy array of shape (N,) with similarity probabilities.
        """
        if not candidates:
            return np.array([])

        # 1. Prepare Track Data (Query)
        # Use robust ID (Slow Memory) if available, else Fast Memory
        t_feat = track.robust_id if track.robust_id is not None else track.last_known_feature
        if t_feat is None:
            return np.zeros(len(candidates))

        # BUGFIX: Validate and enforce feature normalization
        feat_norm = np.linalg.norm(t_feat)
        if abs(feat_norm - 1.0) > 0.1:  # Allow small numerical errors
            print(f"[GCN] WARNING: Track feature not normalized (||f||={feat_norm:.4f}), fixing...")
            t_feat = t_feat / (feat_norm + 1e-8)

        # Get track's camera resolution (use stored value or fallback)
        t_w, t_h = getattr(track, 'last_cam_res', (1920, 1080))

        # Normalize track bbox using its original camera resolution
        t_bbox_norm = self._normalize_bbox(track.last_seen_bbox, t_w, t_h)

        # Calculate normalized time gap
        dt = curr_time - track.last_seen_timestamp
        norm_dt = np.tanh(dt / 10.0)  # Squash to [-1, 1] range

        # FIXED: Create 5D geometry vector (4 bbox + 1 time)
        t_geo = np.concatenate([t_bbox_norm, [norm_dt]])

        # Combine features: 1024 (appearance) + 5 (geometry+time) = 1029 dimensions
        t_input = np.concatenate([t_feat, t_geo])
        
        # 2. Prepare Candidates Data (Keys)
        d_inputs = []
        for cand in candidates:
            d_feat = cand['feature']

            # BUGFIX: Validate and enforce feature normalization for candidates
            feat_norm = np.linalg.norm(d_feat)
            if abs(feat_norm - 1.0) > 0.1:
                print(f"[GCN] WARNING: Candidate feature not normalized (||f||={feat_norm:.4f}), fixing...")
                d_feat = d_feat / (feat_norm + 1e-8)

            d_bbox_norm = self._normalize_bbox(cand['bbox'], frame_w, frame_h)

            # Candidates are current detections, so dt=0
            # FIXED: Add time dimension (5D geometry: 4 bbox + 1 time)
            d_geo = np.concatenate([d_bbox_norm, [0.0]])

            d_inputs.append(np.concatenate([d_feat, d_geo]))

        # 3. Batch Tensor Creation
        # Shape: (1, 1029, 1) -> Batch=1, Dim=1029 (1024 feat + 5 geo), N=1 Track
        t_tensor = torch.tensor(t_input, dtype=torch.float32).to(self.device)
        t_tensor = t_tensor.unsqueeze(0).unsqueeze(-1)

        # Shape: (1, 1029, M) -> Batch=1, Dim=1029, M Candidates
        d_stack = np.stack(d_inputs, axis=1)  # (1029, M)
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