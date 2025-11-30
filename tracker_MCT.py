# tracker_MCT.py
# FINAL FIXED VERSION - DUAL-QUERY MATCHING IMPLEMENTED
#
# CHANGES:
# 1. Implemented Dual-Query (Fast vs Slow) matching in _fast_match and _cross_camera_match.
# 2. Added 'robust_id' attribute to GlobalTrack for Slow Memory storage.
# 3. Helper function 'cosine_distance_single' added for point-to-point comparison.
# 4. BUGFIX: Added missing 'last_known_feature' attribute to GlobalTrack.__init__

import numpy as np
from scipy.signal import savgol_filter 
from numpy.linalg import inv, norm 
import pickle 
import time 
import faiss 
from datetime import datetime 
from staff_filter import StaffFilter
import threading

try:
    from gcn_handler import GCNHandler
    GCN_AVAILABLE = True
except ImportError:
    print("WARNING: 'gcn_handler.py' not found! GCN disabled.")
    GCN_AVAILABLE = False

try:
    from re_ranking import re_ranking
    RERANK_AVAILABLE = True
except ImportError:
    print("WARNING: 're_ranking.py' not found! Re-Ranking disabled.")
    RERANK_AVAILABLE = False
    def re_ranking(q, g, **kwargs): return None

def get_direction(v):
    m = norm(v)
    return (v / (m + 1e-6), m) if m > 0 else (np.zeros_like(v), 0)

def calculate_iou(bbox1, bbox2):
    """Intersection over Union (IoU)"""
    xx1 = max(bbox1[0], bbox2[0]); yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2]); yy2 = min(bbox1[3], bbox2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter_area = w * h
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-6)

def cosine_distance(query_vec, gallery_vecs):
    """Fast cosine distance using numpy."""
    if len(gallery_vecs) == 0:
        return np.array([])
    similarities = gallery_vecs @ query_vec.T
    distances = 1.0 - similarities.squeeze()
    if distances.ndim == 0:
        distances = np.array([distances])
    return distances

def cosine_distance_single(query, target):
    """Computes distance between 1 query and 1 target (0..2)."""
    if query is None or target is None: return 2.0
    # Assuming inputs are already normalized
    sim = np.dot(query, target)
    return 1.0 - sim

def compute_fused_distance(query_vec, query_bbox, gallery_vecs, gallery_bboxes, 
                           same_camera_mask, alpha=0.7, beta=0.3):
    """Compute fused distance combining appearance and geometry."""
    N = len(gallery_vecs)
    if N == 0:
        return np.array([])
    
    app_dist = cosine_distance(query_vec, gallery_vecs)
    
    iou_dist = np.ones(N)
    for i, (gb, same_cam) in enumerate(zip(gallery_bboxes, same_camera_mask)):
        if same_cam and gb is not None:
            iou_val = calculate_iou(query_bbox, gb)
            iou_dist[i] = 1.0 - iou_val
    
    fused = np.where(same_camera_mask, alpha * app_dist + beta * iou_dist, app_dist)
    return fused


class OCSORTTracker:
    def __init__(self, dt, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
        self.dt = dt
        self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = process_noise_cov 
        self.R_base = measurement_noise_cov
        self.x = initial_state      
        self.P = initial_covariance 
        self.last_observation = initial_state[:2]
        self.time_since_update = 0
        self.history_observations = []; self.history_x = []; self.history_y = []
        self.smooth_pos = initial_state[:2]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += self.dt
        return self.x

    def update(self, z, current_bbox=None, last_bbox=None, use_orb=True, conf=None):
        adaptive_R = self.R_base.copy()
        
        if conf is not None:
            scale_factor = 1.0 / (conf + 1e-4)
            adaptive_R *= scale_factor

        if current_bbox is not None and last_bbox is not None:
            h_curr = current_bbox[3] - current_bbox[1]
            h_last = last_bbox[3] - last_bbox[1]
            if h_last > 0:
                ratio = (h_curr - h_last) / h_last
                if ratio > 0.40 or ratio < -0.25: adaptive_R *= 10.0
        
        if use_orb and self.time_since_update > (self.dt * 3.0):
            delta_pos = z - self.last_observation
            delta_time = self.time_since_update
            virtual_velocity = delta_pos / (delta_time + 1e-6)
            prev_speed = norm(self.x[2:])
            curr_speed = norm(virtual_velocity)
            if prev_speed > 0 and curr_speed > (prev_speed * 3.0): virtual_velocity *= 0.5
            self.x = np.array([z[0], z[1], virtual_velocity[0], virtual_velocity[1]])
            self.P = np.eye(4) * adaptive_R[0,0] * 5.0 

        y = z - self.H @ self.x 
        S = self.H @ self.P @ self.H.T + adaptive_R 
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
        self.last_observation = z
        self.time_since_update = 0
        self.history_observations.append(z)
        if len(self.history_observations) > 10: self.history_observations.pop(0)
        
        self.history_x.append(float(self.x[0])); self.history_y.append(float(self.x[1]))
        if len(self.history_x) > 7: self.history_x.pop(0); self.history_y.pop(0)
        
        if len(self.history_x) >= 7:
            try:
                self.smooth_pos = np.array([savgol_filter(self.history_x, 7, 2)[-1], savgol_filter(self.history_y, 7, 2)[-1]])
            except (ValueError, IndexError) as e:
                # FIXED: Catch specific exceptions, fallback to raw position
                self.smooth_pos = self.x[:2]
        else:
            self.smooth_pos = self.x[:2]
        return self.x

    def interpolate(self, new_pos, n_steps):
        if n_steps <= 0: return
        start_pos = self.last_observation
        step_vec = (new_pos - start_pos) / (n_steps + 1)
        for i in range(1, n_steps + 1):
            inter_p = start_pos + (step_vec * i)
            self.history_x.append(float(inter_p[0]))
            self.history_y.append(float(inter_p[1]))
            if len(self.history_x) > 15:
                self.history_x.pop(0); self.history_y.pop(0)

# --- NEW: Uncertainty Adjusted Distance ---
def uncertainty_adjusted_distance(query, target_mean, target_var):
    """
    Computes Cosine Distance, relaxed by the target's Variance (Uncertainty).
    Formula: Dist_Adj = Dist_Cos / (1.0 + Scale * Mean_Variance)
    """
    if query is None or target_mean is None: return 2.0
    
    # 1. Standard Cosine Distance (0.0 to 2.0)
    sim = np.dot(query, target_mean)
    raw_dist = 1.0 - sim
    
    if target_var is None:
        return raw_dist
        
    # 2. Calculate Uncertainty Scalar (Mean of variance vector)
    # You could also use sum() or max(), but mean is stable.
    uncertainty = np.mean(target_var)
    
    # 3. Apply Relaxation
    # Scale Factor (5.0) determines how much uncertainty helps.
    # If uncertainty is 0.05 (high), divisor becomes 1.25, reducing dist by 20%.
    SCALE_FACTOR = 5.0
    adjusted_dist = raw_dist / (1.0 + (uncertainty * SCALE_FACTOR))
    
    return adjusted_dist


class GlobalTrack:
    """
    Represents a globally tracked identity across multiple cameras.
    
    Attributes:
        global_id: Unique identifier for this track
        group_id: Group/location identifier (e.g., "mall_1")
        kf: Kalman filter for position estimation
        last_cam_id: Last camera where this track was seen
        last_seen_bbox: Last bounding box coordinates [x1, y1, x2, y2]
        last_seen_timestamp: Unix timestamp of last observation
        is_staff: Whether this track is identified as staff
        shadow_role: Staff role if is_staff is True
        shadow_name: Staff name if is_staff is True
        
        # Memory States (Dual-Query System):
        last_known_feature: Most recent feature vector (Fast Memory - immediate)
        fast_buffer: List of recent features for robust matching
        robust_id: Slow Memory mean vector (stable identity from ContinuumState)
        robust_var: Slow Memory variance (uncertainty estimation)
    """
    def __init__(self, global_id, group_id, dt, initial_state, initial_P, Q_cov, R_cov):
        self.global_id = global_id
        self.group_id = group_id
        self.kf = OCSORTTracker(dt, initial_state, initial_P, Q_cov, R_cov)
        self.last_cam_id = None
        self.last_seen_bbox = [0, 0, 0, 0]
        self.last_seen_timestamp = time.time()
        self.is_staff = False
        self.shadow_role = None
        self.shadow_name = None
        self.last_cam_res = (1920, 1080)
        
        # --- MEMORY STATES (Dual-Query System) ---
        # Fast Memory: Immediate appearance features
        self.last_known_feature = None  # BUGFIX: This was missing!
        self.fast_buffer = []           # Buffer of recent features for matching
        
        # Slow Memory: Stable identity from Continuum Memory System
        self.robust_id = None           # Mean vector (consolidated identity)
        self.robust_var = None          # Variance vector (uncertainty)


class TrackerManagerMCT:
    def __init__(self, dt, Q_cov, R_cov, feature_dim=1024, redis_client=None, topology_manager=None):
        self.redis_client = redis_client; self.topology = topology_manager
        self.feature_dim = feature_dim; self.dt = dt
        self.Q_cov = Q_cov; self.R_cov = R_cov
        
        # Relaxed Threshold slightly to allow Dual-Query flexibility
        self.reid_threshold = 0.55
        self.max_time_gap = 60.0
        self.staff_filters = {}; self.global_tracks = {}
        self.edge_to_global_map = {}; self.pending_tracks = {}
        self.last_gc_time = time.time(); self.gc_interval = 60.0
        self.max_keep_alive = 300.0; self.max_faiss_size = 20000
        self.faiss_index = faiss.IndexFlatIP(self.feature_dim); self.faiss_id_map = []
        
        self.use_reranking_cross_camera = True
        self.use_reranking_same_camera = False
        self.rerank_cache = {}
        self.rerank_cache_ttl = 5.0
        
        self.min_features_for_id = 3

        self._faiss_lock = threading.RLock()
        
        self.gcn_refiner = None
        if GCN_AVAILABLE:
            try:
                self.gcn_refiner = GCNHandler("models/sota_gcn_dinov3_5_2dim.pth")
                print("[Central] GCN Refiner Loaded.")
            except Exception as e:
                print(f"[Central] GCN Load Failed: {e}")

        if not self.redis_client.exists("mct:next_global_id"):
            self.redis_client.set("mct:next_global_id", 1)

    def _get_staff_filter(self, group_id):
        if group_id not in self.staff_filters:
            self.staff_filters[group_id] = StaffFilter(self.redis_client, group_id)
        return self.staff_filters[group_id]

    def _run_garbage_collection(self):
        with self._faiss_lock:
            curr_time = time.time()
            expired = [gid for gid, t in self.global_tracks.items() if (curr_time - t.last_seen_timestamp) > self.max_keep_alive]
            if expired:
                print(f"[GC] Removing {len(expired)} expired tracks.")
                for gid in expired:
                    del self.global_tracks[gid]
                    keys_to_del = [k for k, v in self.edge_to_global_map.items() if v == gid]
                    for k in keys_to_del: del self.edge_to_global_map[k]
        
        if self.faiss_index.ntotal > self.max_faiss_size:
            print("[GC] Rebuilding FAISS Index...")
            active_gids = list(self.global_tracks.keys())
            new_vectors = []
            new_id_map = []
            
            for gid in active_gids:
                track = self.global_tracks[gid]
                # Use Robust ID if available, else Fast ID (last_known_feature)
                # FIXED: Check if both are None before processing
                feat = track.robust_id if track.robust_id is not None else track.last_known_feature
                if feat is None:
                    continue  # Skip tracks without any feature vector

                norm_feat = feat / (norm(feat) + 1e-6)
                new_vectors.append(norm_feat.astype(np.float32))
                new_id_map.append(gid)
            
            self.faiss_index.reset()
            self.faiss_id_map = []
            
            if new_vectors:
                vectors_array = np.array(new_vectors).astype(np.float32)
                self.faiss_index.add(vectors_array)
                self.faiss_id_map = new_id_map
            
            print(f"[GC] FAISS rebuilt: {len(new_id_map)} vectors")
        
        expired_cache = [k for k, v in self.rerank_cache.items() 
                        if curr_time - v['time'] > self.rerank_cache_ttl]
        for k in expired_cache:
            del self.rerank_cache[k]

    def _manage_gallery_diversity(self, gid, new_vector):
        if new_vector is None: return None
        key = f"gallery_core:{gid}"
        norm_vec = new_vector / (norm(new_vector) + 1e-6)
        
        data = self.redis_client.get(key)
        core_set = pickle.loads(data) if data else []
        
        if not core_set:
            core_set.append(norm_vec)
        else:
            dists = [np.dot(v, norm_vec) for v in core_set]
            max_sim = max(dists)
            best_idx = np.argmax(dists)
            
            if max_sim > 0.95:
                core_set[best_idx] = 0.9 * core_set[best_idx] + 0.1 * norm_vec
                core_set[best_idx] /= (norm(core_set[best_idx]) + 1e-6)
            elif max_sim < 0.85 and len(core_set) < 5:
                core_set.append(norm_vec)
            elif len(core_set) >= 5:
                core_set[best_idx] = 0.7 * core_set[best_idx] + 0.3 * norm_vec
                core_set[best_idx] /= (norm(core_set[best_idx]) + 1e-6)

        self.redis_client.set(key, pickle.dumps(core_set), ex=86400)
        return norm_vec

    def _update_counters(self, cam_id, group_id, mode, gid=None, role=None):
        today = datetime.now().strftime("%Y-%m-%d")
        p = self.redis_client.pipeline()
        if role: 
            key = f"mct:shadow:{group_id}:{today}"
            p.hincrby(key, f"{role}_count", 1)
        else:
            if mode == "tripwire":
                p.hincrby(f"mct:stats:{cam_id}:{today}", "total_tripwire", 1)
            elif mode == "unique" and gid:
                p.pfadd(f"mct:unique:{cam_id}:{today}", gid)
        p.execute()

    def _calculate_ocm_cost(self, track, new_gp, dt):
        track_vel = track.kf.x[2:] 
        candidate_vec = new_gp - track.kf.last_observation
        candidate_vel = candidate_vec / (dt + 1e-6)
        t_dir, t_mag = get_direction(track_vel)
        c_dir, c_mag = get_direction(candidate_vel)
        cos_sim = np.dot(t_dir, c_dir)
        if t_mag < 0.5: return 0.0 
        return 0.5 * (1.0 - cos_sim)

    # ------------------------------------------------------------
    # MODIFIED: _fast_match with DUAL-QUERY Logic
    # ------------------------------------------------------------
    def _fast_match(self, cam_id, group_id, feature, bbox, gp, curr_time, frame_res=(1920, 1080)):
        if self.faiss_index.ntotal == 0:
            return None, 100.0
        
        q_vec = np.array([feature]).astype(np.float32)
        q_vec = q_vec / (norm(q_vec) + 1e-6)
        
        shortlist_k = min(20, self.faiss_index.ntotal)
        # FIXED: Use snapshot to avoid race condition during FAISS access
        with self._faiss_lock:
            D_raw, I_raw = self.faiss_index.search(q_vec, k=shortlist_k)
            id_map_snapshot = self.faiss_id_map.copy()

        candidates = []
        for idx in I_raw[0]:
            if idx == -1 or idx >= len(id_map_snapshot): continue
            cand_gid = id_map_snapshot[idx]  # Use snapshot instead of self.faiss_id_map
            cand_track = self.global_tracks.get(cand_gid)
            if cand_track and cand_track.group_id == group_id:
                candidates.append(cand_track)
        
        if not candidates:
            return None, 100.0
        
        best_gid = None
        best_score = 100.0
        
        for track in candidates:
            # DUAL-QUERY: Compare against Fast Memory (Instant) AND Slow Memory (Robust)
            if track.fast_buffer:
                # Calculate distance to ALL vectors in the buffer
                # This acts like "Max-Pooling" attention mechanism
                dists = [cosine_distance_single(q_vec[0], f) for f in track.fast_buffer]
                dist_fast = min(dists) # Take the best angle
            else:
                # Fallback to last_known_feature if buffer is empty
                dist_fast = cosine_distance_single(q_vec[0], track.last_known_feature)
            
            dist_slow = uncertainty_adjusted_distance(
                q_vec[0], 
                track.robust_id, 
                track.robust_var
            )
            
            # Take the BEST match (Minimum distance)
            app_dist = min(dist_fast, dist_slow)
            
            # Fuse IoU
            if track.last_cam_id == cam_id:
                iou_val = calculate_iou(bbox, track.last_seen_bbox)
                iou_dist = 1.0 - iou_val
                fused = 0.7 * app_dist + 0.3 * iou_dist
            else:
                fused = app_dist
            
            # Topology Check
            dt = curr_time - track.last_seen_timestamp
            if self.topology and track.last_cam_id != cam_id:
                prob_geo = self.topology.get_transition_prob(group_id, track.last_cam_id, cam_id, dt)
                if prob_geo < 0.01: fused = 100.0
                elif prob_geo < 0.5: fused *= 1.2
            
            motion_cost = self._calculate_ocm_cost(track, gp, dt)
            if motion_cost > 0.7: fused = 100.0
            elif motion_cost > 0.3: fused *= 1.15
            
            # Conflict Check
            for k, v in self.edge_to_global_map.items():
                if v == track.global_id and k.startswith(f"{cam_id}_"):
                    fused = 100.0
                    break
        
            if fused < self.reid_threshold and fused < best_score:
                best_score = fused
                best_gid = track.global_id
        
        return best_gid, best_score

    # ------------------------------------------------------------
    # IMPROVED: _cross_camera_match with BATCH TRANSFORMER
    # ------------------------------------------------------------
    def _cross_camera_match(self, cam_id, group_id, feature, bbox, gp, curr_time, frame_res):
        """
        Optimized matching using Batch Inference for the Transformer.
        Requires frame dimensions for correct normalization.
        """
        if self.faiss_index.ntotal == 0:
            return None, 100.0
        
        # Unpack current camera resolution
        frame_w, frame_h = frame_res
        
        q_vec = np.array([feature]).astype(np.float32)
        q_vec = q_vec / (norm(q_vec) + 1e-6)

        # 1. Candidate Retrieval
        shortlist_k = min(100, self.faiss_index.ntotal)
        
        with self._faiss_lock:
            D_raw, I_raw = self.faiss_index.search(q_vec, k=shortlist_k)
            id_map_snapshot = self.faiss_id_map.copy()

        # 2. Pre-filtering & Batch Preparation
        batch_candidates = []
        VISUAL_CUTOFF = 0.7 
        
        for idx in I_raw[0]:
            if idx == -1 or idx >= len(id_map_snapshot): continue
            
            cand_gid = id_map_snapshot[idx]
            track = self.global_tracks.get(cand_gid)
            
            # Validity Checks
            if not track or track.group_id != group_id: continue
            if track.last_cam_id == cam_id: continue
            
            dt = curr_time - track.last_seen_timestamp
            if dt > self.max_time_gap: continue

            # Appearance Score
            dist_fast = cosine_distance_single(q_vec[0], track.last_known_feature)
            dist_slow = uncertainty_adjusted_distance(q_vec[0], track.robust_id, track.robust_var)
            visual_score = min(dist_fast, dist_slow)
            
            # Topology & Motion Logic
            geo_penalty = 1.0
            if self.topology:
                prob_geo = self.topology.get_transition_prob(group_id, track.last_cam_id, cam_id, dt)
                if prob_geo < 0.01: continue 
                elif prob_geo < 0.5: geo_penalty = 1.2
            
            motion_cost = self._calculate_ocm_cost(track, gp, dt)
            if motion_cost > 0.7: continue
            elif motion_cost > 0.3: geo_penalty *= 1.15
            
            final_visual_score = visual_score * geo_penalty

            # Add to batch if score is reasonable (even if slightly bad visually)
            if final_visual_score < VISUAL_CUTOFF:
                batch_candidates.append({
                    'gid': cand_gid,
                    'track': track,
                    'visual_score': final_visual_score
                })

        # 3. Batch Transformer Refinement
        best_gid = None
        best_score = 100.0

        if self.gcn_refiner and batch_candidates:
            try:
                # --- STRATEGY: PRE-NORMALIZATION ---
                # We normalize everything to [0, 1] manually here.
                # Then we pass 1.0 as resolution to the refiner to bypass its internal normalization.
                
                # A. Prepare Query (Current Detection)
                # Normalize using CURRENT camera resolution
                norm_q_bbox = [
                    bbox[0] / frame_w, bbox[1] / frame_h,
                    bbox[2] / frame_w, bbox[3] / frame_h
                ]

                # Create Dummy Track for the Query (Refiner expects an object)
                class DummyTrack:
                    def __init__(self, feat, box, timestamp):
                        self.robust_id = feat
                        self.last_known_feature = feat
                        self.last_seen_bbox = box # Pre-normalized
                        self.last_cam_res = (1.0, 1.0)
                        self.last_seen_timestamp = timestamp  # BUGFIX: Required by gcn_handler

                dummy_query = DummyTrack(feature, norm_q_bbox, curr_time)
                
                # B. Prepare Candidates (Past Global Tracks)
                refiner_candidates = []
                for item in batch_candidates:
                    t = item['track']
                    feat = t.robust_id if t.robust_id is not None else t.last_known_feature
                    
                    # Get track's original resolution (fallback to 1920x1080 if missing)
                    t_w, t_h = getattr(t, 'last_cam_res', (1920, 1080))
                    
                    # Normalize track bbox using ITS OWN historical resolution
                    t_bbox = t.last_seen_bbox
                    norm_t_bbox = [
                        t_bbox[0] / t_w, t_bbox[1] / t_h,
                        t_bbox[2] / t_w, t_bbox[3] / t_h
                    ]
                    
                    refiner_candidates.append({
                        'feature': feat,
                        'bbox': norm_t_bbox # Pre-normalized
                    })
                
                # C. Run Batch Inference
                # Pass frame_w=1.0, frame_h=1.0 because inputs are already normalized [0, 1]
                refine_scores = self.gcn_refiner.predict_batch(
                    dummy_query, 
                    refiner_candidates, 
                    frame_w=1.0, 
                    frame_h=1.0,
                    curr_time=curr_time
                )
                
                # D. Fuse Scores
                for i, item in enumerate(batch_candidates):
                    visual = item['visual_score']
                    gcn_conf = refine_scores[i] # Higher value = Better Match
                    gcn_dist = 1.0 - gcn_conf
                    
                    # Fusion: 60% Visual, 40% Relation/Geometry
                    fused_score = visual * 0.6 + gcn_dist * 0.4
                    
                    if fused_score < self.reid_threshold and fused_score < best_score:
                        best_score = fused_score
                        best_gid = item['gid']
                        
            except Exception as e:
                print(f"[Tracker] Refinement Error: {e}")
                # Fallback to best visual score if GCN fails
                for item in batch_candidates:
                    if item['visual_score'] < best_score:
                        best_score = item['visual_score']
                        best_gid = item['gid']

        else:
            # No refiner available, pick best visual
            for item in batch_candidates:
                if item['visual_score'] < best_score:
                    best_score = item['visual_score']
                    best_gid = item['gid']

        return best_gid, best_score

    def update_edge_track_position(self, cam_id, group_id, edge_id, gp, conf, bbox):
        map_key = f"{cam_id}_{edge_id}"
        gid = self.edge_to_global_map.get(map_key)
        
        if gid and gid in self.global_tracks:
            track = self.global_tracks[gid]
            track.kf.predict()
            track.kf.update(gp, current_bbox=bbox, last_bbox=track.last_seen_bbox, use_orb=False, conf=conf)
            track.last_seen_bbox = list(bbox) if not isinstance(bbox, list) else bbox
            track.last_seen_timestamp = time.time()
            track.last_cam_id = cam_id
            
            sf = self._get_staff_filter(group_id)
            is_staff, role, name = sf.identify_staff(vector=None, global_id=gid)
            if is_staff:
                track.is_staff = True; track.shadow_role = role; track.shadow_name = name
            return track
        elif map_key in self.pending_tracks:
            self.pending_tracks[map_key]["kf"].predict()
            self.pending_tracks[map_key]["kf"].update(gp, current_bbox=bbox, last_bbox=self.pending_tracks[map_key]["last_bbox"], use_orb=False, conf=conf)
            self.pending_tracks[map_key]["last_bbox"] = list(bbox) if not isinstance(bbox, list) else bbox
        return None

    def update_edge_track_feature(self, cam_id, group_id, edge_id, gp, conf, bbox, feature, quality_score, frame_res=(1920, 1080)):
        if time.time() - self.last_gc_time > self.gc_interval:
            self._run_garbage_collection()
            self.last_gc_time = time.time()

        map_key = f"{cam_id}_{edge_id}"
        curr_time = time.time()
        bbox = list(bbox) if not isinstance(bbox, list) else bbox
        
        existing_gid = self.edge_to_global_map.get(map_key)
        
        if existing_gid and existing_gid in self.global_tracks:
            best_gid = existing_gid
        else:
            best_gid, best_score = self._fast_match(cam_id, group_id, feature, bbox, gp, curr_time, frame_res)
            if best_gid is None:
                best_gid, best_score = self._cross_camera_match(cam_id, group_id, feature, bbox, gp, curr_time, frame_res)

        if best_gid:
            gt = self.global_tracks[best_gid]
            sf = self._get_staff_filter(group_id)
            is_staff, role, name = sf.identify_staff(vector=feature, global_id=best_gid)
            if is_staff:
                gt.is_staff = True; gt.shadow_role = role; gt.shadow_name = name

            if self.topology and gt.last_cam_id and gt.last_cam_id != cam_id:
                self.topology.update_topology(group_id, gt.last_cam_id, cam_id, curr_time - gt.last_seen_timestamp)
            
            if gt.last_cam_id == cam_id:
                time_gap = curr_time - gt.last_seen_timestamp
                if 0.2 < time_gap < 2.0:
                    steps = int(time_gap * 20) 
                    if steps > 0: gt.kf.interpolate(gp, steps)

            gt.kf.update(gp, current_bbox=bbox, last_bbox=gt.last_seen_bbox, use_orb=True, conf=conf)
            
            is_bad_quality_box = False
            if gt.last_seen_bbox is not None:
                h_curr = bbox[3] - bbox[1]; h_last = gt.last_seen_bbox[3] - gt.last_seen_bbox[1]
                if h_last > 0:
                    ratio = (h_curr - h_last) / h_last
                    if ratio < -0.25 or ratio > 0.40: is_bad_quality_box = True
            
            # Update Fast Buffer
            gt.fast_buffer.append(feature)
            if len(gt.fast_buffer) > 5: # Keep last 5
                gt.fast_buffer.pop(0)
            
            gt.last_seen_bbox = bbox
            gt.last_seen_timestamp = curr_time
            gt.last_cam_id = cam_id
            
            # --- UPDATE FAST MEMORY ---
            gt.last_known_feature = feature
            # Note: gt.robust_id is NOT updated here; it is updated via Central Service -> Continuum Memory

            gt.last_cam_res = frame_res
            
            self.edge_to_global_map[map_key] = best_gid
            if map_key in self.pending_tracks: del self.pending_tracks[map_key]
            
            if not is_bad_quality_box:
                self._manage_gallery_diversity(best_gid, feature)
            
            if not gt.is_staff: self._update_counters(cam_id, group_id, "unique", best_gid)
            return gt
        
        else:
            if map_key in self.pending_tracks:
                pt = self.pending_tracks[map_key]
                pt["kf"].update(gp, current_bbox=bbox, last_bbox=pt["last_bbox"], use_orb=False, conf=conf)
                pt["features_count"] += 1
                pt["last_bbox"] = bbox
                
                if "first_feature" not in pt:
                    pt["first_feature"] = feature
                pt["last_feature"] = feature
                
                if pt["features_count"] >= self.min_features_for_id:
                    pt_data = self.pending_tracks.pop(map_key)
                    new_gid = int(self.redis_client.incr("mct:next_global_id"))
                    
                    new_gt = GlobalTrack(
                        new_gid, group_id, self.dt, 
                        pt_data["kf"].x, 
                        np.eye(4)*100, 
                        self.Q_cov, 
                        self.R_cov
                    )
                    
                    new_gt.last_seen_bbox = pt_data["last_bbox"]
                    new_gt.last_seen_timestamp = curr_time
                    new_gt.last_cam_id = cam_id
                    new_gt.last_known_feature = pt_data.get("last_feature", feature)
                    # robust_id remains None initially, until CMS processes enough data
                    
                    sf = self._get_staff_filter(group_id)
                    is_staff, role, name = sf.identify_staff(vector=feature, global_id=new_gid)
                    if is_staff:
                        new_gt.is_staff = True
                        new_gt.shadow_role = role
                        new_gt.shadow_name = name
                        self._update_counters(cam_id, group_id, "shadow", role=role)
                    else:
                        self._update_counters(cam_id, group_id, "unique", new_gid)
                    
                    self.global_tracks[new_gid] = new_gt
                    self.edge_to_global_map[map_key] = new_gid
                    
                    norm_vec = self._manage_gallery_diversity(new_gid, feature)
                    
                    if self.faiss_index is not None and norm_vec is not None:
                        self.faiss_index.add(np.array([norm_vec]).astype(np.float32))
                        self.faiss_id_map.append(new_gid)
                    
                    print(f"[Tracker] NEW ID assigned: G{new_gid} for {map_key}")
                    return new_gt
            else:
                kf = OCSORTTracker(
                    self.dt, 
                    np.array([gp[0], gp[1], 0, 0]), 
                    np.eye(4)*100, 
                    self.Q_cov, 
                    self.R_cov
                )
                kf.update(gp, current_bbox=bbox, last_bbox=bbox, use_orb=False, conf=conf)
                
                self.pending_tracks[map_key] = {
                    "kf": kf, 
                    "last_bbox": bbox, 
                    "features_count": 1,
                    "first_feature": feature,
                    "last_feature": feature,
                    "group_id": group_id
                }
            
            return None

    def register_new_edge_track(self, cam_id, group_id, edge_id, gp, conf, bbox):
        map_key = f"{cam_id}_{edge_id}"
        if map_key in self.edge_to_global_map or map_key in self.pending_tracks:
            return
        
        bbox = list(bbox) if not isinstance(bbox, list) else bbox
        
        kf = OCSORTTracker(
            self.dt, 
            np.array([gp[0], gp[1], 0, 0]), 
            np.eye(4)*100, 
            self.Q_cov, 
            self.R_cov
        )
        kf.update(gp, current_bbox=bbox, last_bbox=bbox, use_orb=False, conf=conf)
        
        self.pending_tracks[map_key] = {
            "kf": kf, 
            "last_bbox": bbox, 
            "features_count": 0,
            "group_id": group_id
        }

    def lost_edge_track(self, cam_id, edge_id):
        map_key = f"{cam_id}_{edge_id}"
        self.edge_to_global_map.pop(map_key, None)
        self.pending_tracks.pop(map_key, None)

    def get_global_id_for_edge_track(self, cam_id, edge_id):
        return self.edge_to_global_map.get(f"{cam_id}_{edge_id}", "Unknown")
    
    def get_viz_data_for_camera(self, cam_id):
        viz = []
        for k, gid in self.edge_to_global_map.items():
            if k.startswith(f"{cam_id}_"):
                gt = self.global_tracks.get(gid)
                if gt:
                    viz.append({
                        "edge_track_id": int(k.split('_')[-1]),
                        "global_id": gid,
                        "bbox": gt.last_seen_bbox,
                        "smooth_gp": gt.kf.smooth_pos.tolist(),
                        "is_staff": gt.is_staff,
                        "role": gt.shadow_role,
                        "name": getattr(gt, 'shadow_name', 'Unknown')
                    })
        return viz