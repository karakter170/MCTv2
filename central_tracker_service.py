# central_tracker_service.py
# P2: NESTED LEARNING INTEGRATED VERSION (DUAL-QUERY UPDATE)
#
# CHANGES:
# 1. Updates GlobalTrack.robust_id with Continuum Memory output (instead of last_known_feature).
# 2. Passes quality score to Neural Module for weighted learning.

import redis
import json
import time
import numpy as np
import csv 
import threading
import queue
import signal
import sys

from tracker_MCT import TrackerManagerMCT 
from topology_manager import TopologyManager
from continuum_memory import ContinuumState

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


# --- SETUP ---
r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

topo_manager = TopologyManager(r_bytes)

print("[Central] Tracker Manager Starting (DINOv3 + Nested Learning Mode)...")
tracker = TrackerManagerMCT(
    dt=1.0, 
    Q_cov=np.eye(4)*0.5, 
    R_cov=np.eye(2)*50, 
    feature_dim=1024,
    redis_client=r_bytes, 
    topology_manager=topo_manager
)

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
    Includes Dual-State update logic.
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
            # NESTED LEARNING INTEGRATION (Deep Optimization)
            # ====================================================
            if track and feat is not None:
                try:
                    gid = track.global_id
                    key = f"mct:continuum:{gid}"
                    
                    # A. Load Neural Module State
                    raw_state = r_json.get(key)
                    if raw_state:
                        state_dict = json.loads(raw_state)
                        cms = ContinuumState(data=state_dict)
                    else:
                        cms = ContinuumState() 
                    
                    # B. The Optimization Step (Learns from current feature)
                    cms.learn(feat, quality=quality)
                    
                    # C. Save State Back to Redis
                    r_json.set(key, json.dumps(cms.to_dict()), ex=3600)

                    identity_data = cms.get_identity() # Returns (Mean, Var) or None
                    if identity_data is not None:
                        robust_mean, robust_var = identity_data
                        # Store both on the track for the Tracker to use
                        track.robust_id = robust_mean
                        track.robust_var = robust_var
                    
                    # D. Feedback Loop (DUAL-STATE)
                    # We store the Robust ID separately from the Fast ID
                    robust_identity = cms.get_identity()
                    if robust_identity is not None:
                        track.robust_id = robust_identity
                        
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
        return False


# ============================================
# Stream Consumer Loop
# ============================================
def process_stream():
    """Main processing loop using Redis Streams."""
    print(f"[Stream] Consumer '{CONSUMER_NAME}' starting...")
    retry_counts = {}
    
    while True:
        try:
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
    public_log.stop()
    secret_log.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCT Central Tracker Service')
    parser.add_argument('--mode', choices=['stream', 'pubsub', 'bridge'], 
                       default='stream', help='Processing mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MCT Central Tracker Service")
    print(f"Mode: {args.mode}")
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