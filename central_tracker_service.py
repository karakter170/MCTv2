# central_tracker_service_v2.py
# P2: ENHANCED NESTED LEARNING INTEGRATED VERSION
#
# Uses ContinuumStateV2 with:
# - Quality-weighted learning
# - Temporal decay
# - Multi-modal appearance support
# - Improved breakout mechanism
# - Confidence scoring

import redis
import json
import time
import numpy as np
import csv 
import threading
import queue
import signal
import sys
from typing import Optional

from tracker_MCT import TrackerManagerMCT 
from topology_manager import TopologyManager
from continuum_memory import ContinuumStateV2, ContinuumConfig

# --- CONFIG ---
PUBLIC_CSV = 'MCT_Public_Log.csv'
SECRET_CSV = 'MCT_Shadow_Log_SECRET.csv'

# ============================================
# Redis Streams Configuration
# ============================================
STREAM_NAME = "track_events"           
CONSUMER_GROUP = "mct_processors"      
CONSUMER_NAME = f"processor_{int(time.time())}"
BATCH_SIZE = 100                        
BLOCK_MS = 1000                        
MAX_RETRIES = 3                        
DEAD_LETTER_STREAM = "track_events_dlq"

# ============================================
# Nested Learning Configuration
# ============================================
CONTINUUM_CONFIG = ContinuumConfig(
    buffer_size=7,

    # --- YENİ EKLENEN KISIM ---
    use_learned_gating=True,  # Yapay Zeka kararını aktif et
    gating_model_path="models/gating_network_msmt17.pt", # Eğittiğiniz modelin yolu
    # --------------------------

    alpha_slow_base=0.05,
    alpha_slow_min=0.02,
    alpha_slow_max=0.20,
    stability_thresh=0.65,
    breakout_limit=30,
    breakout_confirmation=10,
    max_modes=3,
    temporal_decay_half_life=30.0,
    bootstrap_frames=15,
    maturity_frames=100,
    min_quality_for_update=0.3
)

# Cache TTL for continuum states in Redis (1 hour)
CONTINUUM_CACHE_TTL = 3600
# ============================================


class BatchCSVWriter:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.file = None
        self.writer = None
        self._running = True
        
    def start(self):
        self.file = open(self.filename, mode='w', newline='', buffering=1) 
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        self.thread.start()

    def write(self, row):
        self.queue.put(row)

    def stop(self):
        self._running = False
        self.queue.put(None) 
        self.thread.join(timeout=5)
        if self.file:
            self.file.close()

    def _worker(self):
        batch = []
        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                batch.append(item)
                if len(batch) >= 50 or self.queue.empty():
                    if self.writer:
                        self.writer.writerows(batch)
                    batch = []
            except queue.Empty:
                if batch and self.writer:
                    self.writer.writerows(batch)
                    batch = []
            except Exception as e:
                print(f"[CSVWriter] Error: {e}")


class NestedLearningManager:
    """
    Manages ContinuumStateV2 instances for all tracked identities.
    Handles Redis persistence and provides statistics.
    """
    
    def __init__(self, redis_client, config: ContinuumConfig = None):
        self.redis = redis_client
        self.config = config or CONTINUUM_CONFIG
        
        # Local cache to reduce Redis calls
        self.local_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 5.0  # Local cache TTL in seconds
        
        # Statistics
        self.stats = {
            'updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'breakouts': 0,
            'new_modes': 0
        }
    
    def _get_key(self, global_id: int) -> str:
        """Get Redis key for a global ID."""
        return f"mct:continuum:v2:{global_id}"
    
    def _load_state(self, global_id: int) -> ContinuumStateV2:
        """Load state from Redis or create new one."""
        current_time = time.time()
        
        # Check local cache first
        if global_id in self.local_cache:
            cache_age = current_time - self.cache_timestamps.get(global_id, 0)
            if cache_age < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return self.local_cache[global_id]
        
        self.stats['cache_misses'] += 1
        
        # Load from Redis
        key = self._get_key(global_id)
        raw_data = self.redis.get(key)
        
        if raw_data:
            try:
                data = json.loads(raw_data)
                cms = ContinuumStateV2(data=data, config=self.config)
            except (json.JSONDecodeError, Exception) as e:
                print(f"[NestedLearning] Error loading state for {global_id}: {e}")
                cms = ContinuumStateV2(config=self.config)
        else:
            cms = ContinuumStateV2(config=self.config)
        
        # Update local cache
        self.local_cache[global_id] = cms
        self.cache_timestamps[global_id] = current_time
        
        return cms
    
    def _save_state(self, global_id: int, cms: ContinuumStateV2):
        """Save state to Redis."""
        key = self._get_key(global_id)
        data = cms.to_dict()
        self.redis.set(key, json.dumps(data), ex=CONTINUUM_CACHE_TTL)
        
        # Update local cache
        self.local_cache[global_id] = cms
        self.cache_timestamps[global_id] = time.time()
    
    def update(self, global_id: int, feature: np.ndarray, quality: float = 1.0) -> dict:
        """
        Update the nested learning state for a global ID.
        
        Args:
            global_id: Global track ID
            feature: Feature vector from DINOv2
            quality: Quality score (0-1)
            
        Returns:
            Dictionary with update results and identity information
        """
        # Load current state
        cms = self._load_state(global_id)
        
        # Track modes before update
        modes_before = len(cms.modes)
        divergence_before = cms.divergence_counter
        
        # Perform learning step
        learn_result = cms.learn(feature, quality=quality)
        
        # Track statistics
        self.stats['updates'] += 1
        
        if len(cms.modes) > modes_before:
            self.stats['new_modes'] += 1
        
        if divergence_before > cms.config.breakout_limit and cms.divergence_counter == 0:
            self.stats['breakouts'] += 1
        
        # Save updated state
        self._save_state(global_id, cms)
        
        # Get identity information
        identity = cms.get_identity()
        
        return {
            'learn_result': learn_result,
            'identity': identity,
            'all_modes': cms.get_all_modes(),
            'confidence': cms.get_confidence(),
            'statistics': cms.get_statistics()
        }
    
    def get_identity(self, global_id: int) -> Optional[tuple]:
        """Get the primary identity for a global ID."""
        cms = self._load_state(global_id)
        return cms.get_identity()
    
    def get_match_score(self, global_id: int, query_vector: np.ndarray) -> float:
        """Get match score between query and stored identity."""
        cms = self._load_state(global_id)
        return cms.match_score(query_vector)
    
    def get_confidence(self, global_id: int) -> float:
        """Get confidence score for a global ID."""
        cms = self._load_state(global_id)
        return cms.get_confidence()
    
    def get_statistics(self, global_id: int) -> dict:
        """Get detailed statistics for a global ID."""
        cms = self._load_state(global_id)
        return cms.get_statistics()
    
    def get_manager_stats(self) -> dict:
        """Get overall manager statistics."""
        return {
            **self.stats,
            'cached_identities': len(self.local_cache)
        }
    
    def cleanup_cache(self, max_age: float = 60.0):
        """Remove old entries from local cache."""
        current_time = time.time()
        expired = [
            gid for gid, ts in self.cache_timestamps.items()
            if current_time - ts > max_age
        ]
        for gid in expired:
            self.local_cache.pop(gid, None)
            self.cache_timestamps.pop(gid, None)


# --- SETUP ---
r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

topo_manager = TopologyManager(r_bytes)

print("[Central] Tracker Manager Starting (DINOv2 + Enhanced Nested Learning)...")
tracker = TrackerManagerMCT(
    dt=1.0, 
    Q_cov=np.eye(4)*0.5, 
    R_cov=np.eye(2)*50, 
    feature_dim=1024,
    redis_client=r_bytes, 
    topology_manager=topo_manager
)

# Initialize Enhanced Nested Learning Manager
nested_learning = NestedLearningManager(r_json, CONTINUUM_CONFIG)
print(f"[Central] Nested Learning Config: {CONTINUUM_CONFIG}")

public_log = BatchCSVWriter(PUBLIC_CSV, ['Time', 'Group', 'Cam', 'GID', 'Event', 'X', 'Y'])
secret_log = BatchCSVWriter(SECRET_CSV, ['Time', 'Group', 'Cam', 'GID', 'Event', 'X', 'Y', 'Role', 'Name'])
public_log.start()
secret_log.start()


# ============================================
# Stream Setup Functions
# ============================================
def setup_stream():
    try:
        r_json.xgroup_create(
            name=STREAM_NAME,
            groupname=CONSUMER_GROUP,
            id='0',
            mkstream=True
        )
        print(f"[Stream] Created consumer group '{CONSUMER_GROUP}'")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"[Stream] Consumer group '{CONSUMER_GROUP}' already exists")
        else:
            raise

def migrate_from_pubsub():
    def bridge_worker():
        pubsub = r_json.pubsub()
        pubsub.subscribe("track_event_stream")
        print("[Migration] Bridge running: pub/sub -> stream")
        for msg in pubsub.listen():
            if msg['type'] == 'message':
                try:
                    r_json.xadd(STREAM_NAME, {'data': msg['data']}, maxlen=100000)
                except Exception as e:
                    print(f"[Migration] Bridge error: {e}")
    thread = threading.Thread(target=bridge_worker, daemon=True)
    thread.start()
    return thread


def handle_event(data, message_id=None):
    """
    Process a single track event.
    Includes Enhanced Nested Learning with multi-modal support.
    """
    try:
        cam = data['camera_id']
        group = data.get('group_id', 'default')
        evt = data['event_type']
        eid = data['edge_track_id']
        gp = np.array(data['gp_coord']) if data['gp_coord'] else None
        feat = np.array(data['feature']) if data.get('feature') else None
        quality = data.get('quality', 1.0) 
        
        track = None
        
        if evt == "TRACK_NEW": 
            tracker.register_new_edge_track(cam, group, eid, gp, data['conf'], data['bbox'])
            
        elif evt == "TRACK_UPDATE_GP": 
            track = tracker.update_edge_track_position(cam, group, eid, gp, data['conf'], data['bbox'])
            
        elif evt == "TRACK_UPDATE_FEATURE": 
            # 1. Update Spatial/Visual State in Tracker (FAST MEMORY)
            track = tracker.update_edge_track_feature(
                cam, group, eid, gp, data['conf'], 
                data['bbox'], feat, quality
            )
            
            # ====================================================
            # ENHANCED NESTED LEARNING INTEGRATION
            # ====================================================
            if track and feat is not None:
                try:
                    gid = track.global_id
                    
                    # Perform nested learning update
                    result = nested_learning.update(gid, feat, quality=quality)
                    
                    # Extract identity information
                    identity = result['identity']
                    if identity is not None:
                        robust_mean, robust_var = identity
                        track.robust_id = robust_mean
                        track.robust_var = robust_var
                    
                    # Log interesting events
                    learn_result = result['learn_result']
                    
                    if learn_result.get('num_modes', 1) > 1:
                        # Track has multiple appearance modes
                        pass  # Could log this for analysis
                    
                    # Periodic statistics logging (every 100 updates)
                    if nested_learning.stats['updates'] % 100 == 0:
                        print(f"[NestedLearning] Stats: {nested_learning.get_manager_stats()}")
                        
                except Exception as e:
                    print(f"[NestedLearning] Error optimizing track {eid}: {e}")
            # ====================================================

        elif evt == "TRACK_LOST": 
            tracker.lost_edge_track(cam, eid)

        # Logging & Visualization
        if gp is not None and track:
            gx, gy = track.kf.smooth_pos
            gid = track.global_id
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['timestamp']))
            
            if track.is_staff:
                secret_log.write([ts, group, cam, gid, evt, gx, gy, track.shadow_role, "Unknown"])
            else:
                public_log.write([ts, group, cam, gid, evt, gx, gy])

        # Publish visualization update
        viz = tracker.get_viz_data_for_camera(cam)
        if viz:
            r_json.publish(f"results_viz_stream:{cam}", json.dumps(viz))
        
        return True
        
    except Exception as e: 
        print(f"[Handler] Error processing {message_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================
# Stream Consumer Loop
# ============================================
def process_stream():
    """Main processing loop using Redis Streams."""
    print(f"[Stream] Consumer '{CONSUMER_NAME}' starting...")
    retry_counts = {}
    last_cache_cleanup = time.time()
    
    while True:
        try:
            # Periodic cache cleanup
            if time.time() - last_cache_cleanup > 60.0:
                nested_learning.cleanup_cache()
                last_cache_cleanup = time.time()
            
            # Read messages
            messages = r_json.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={STREAM_NAME: '>'},
                count=BATCH_SIZE,
                block=BLOCK_MS
            )
            
            if not messages:
                # Handle pending/failed messages
                pending = r_json.xpending_range(
                    STREAM_NAME, CONSUMER_GROUP,
                    min='-', max='+', count=10
                )
                for p in pending:
                    msg_id = p['message_id']
                    if p['time_since_delivered'] > 30000: # 30s timeout
                        claimed = r_json.xclaim(
                            STREAM_NAME, CONSUMER_GROUP, CONSUMER_NAME,
                            min_idle_time=30000, message_ids=[msg_id]
                        )
                        for cid, cdata in claimed:
                            retry_counts[cid] = retry_counts.get(cid, 0) + 1
                            if retry_counts[cid] > MAX_RETRIES:
                                r_json.xadd(DEAD_LETTER_STREAM, {
                                    'original_id': cid, 
                                    'data': cdata.get('data', '{}'), 
                                    'error': 'max_retries'
                                })
                                r_json.xack(STREAM_NAME, CONSUMER_GROUP, cid)
                            else:
                                data = json.loads(cdata.get('data', '{}'))
                                if handle_event(data, cid):
                                    r_json.xack(STREAM_NAME, CONSUMER_GROUP, cid)
                continue
            
            # Process new messages
            for stream_name, stream_messages in messages:
                for message_id, message_data in stream_messages:
                    try:
                        data_str = message_data.get('data', '{}')
                        data = json.loads(data_str)
                        
                        if handle_event(data, message_id):
                            r_json.xack(STREAM_NAME, CONSUMER_GROUP, message_id)
                    except json.JSONDecodeError:
                        r_json.xack(STREAM_NAME, CONSUMER_GROUP, message_id)
                        
        except redis.ConnectionError as e:
            print(f"[Stream] Redis error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[Stream] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

def process_pubsub_legacy():
    print("[Legacy] Using Pub/Sub mode")
    pubsub = r_json.pubsub()
    pubsub.subscribe("track_event_stream")
    for msg in pubsub.listen():
        if msg['type'] == 'message':
            try:
                data = json.loads(msg['data'])
                handle_event(data)
            except Exception as e:
                print(f"[Legacy] Error: {e}")

# ============================================
# Graceful Shutdown
# ============================================
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    print("\n[Central] Shutdown signal received...")
    shutdown_flag.set()
    
    # Print final statistics
    print(f"[NestedLearning] Final Stats: {nested_learning.get_manager_stats()}")
    
    public_log.stop()
    secret_log.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCT Central Tracker Service (Enhanced)')
    parser.add_argument('--mode', choices=['stream', 'pubsub', 'bridge'], 
                       default='stream', help='Processing mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MCT Central Tracker Service - ENHANCED")
    print(f"Mode: {args.mode}")
    print(f"Nested Learning: ContinuumStateV2 (Multi-Modal)")
    print("=" * 60)
    
    if args.mode == 'stream':
        setup_stream()
        print("[Central] Service Running with Redis Streams...")
        process_stream()
        
    elif args.mode == 'pubsub':
        print("[Central] Service Running with Pub/Sub (legacy)...")
        process_pubsub_legacy()
        
    elif args.mode == 'bridge':
        setup_stream()
        migrate_from_pubsub()
        print("[Central] Bridge mode: forwarding pub/sub to stream...")
        process_stream()