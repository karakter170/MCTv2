# edge_camera.py
# P1 FIXED VERSION - STREAM COMPATIBLE
#
# CHANGES:
# 1. Fixed Protocol Mismatch: Now sends to Redis Streams (xadd) instead of Pub/Sub.
# 2. Added r_json client for proper Stream handling.

import cv2
import time
import json
import redis
import numpy as np
from ultralytics import YOLO
import threading
from collections import OrderedDict

try:
    from trt_loader import TensorRTReidExtractor
except ImportError:
    print("ERROR: 'trt_loader.py' is missing!")
    exit()


# ============================================
# MOTION COMPENSATION (CMC)
# ============================================
class CameraMotionCompensation:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)

    def compute(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = curr_gray
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=100, qualityLevel=0.01, minDistance=30
            )
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix

        if self.prev_keypoints is None or len(self.prev_keypoints) < 20:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=100, qualityLevel=0.01, minDistance=30
            )
            self.prev_frame = curr_gray
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix

        curr_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, curr_gray, self.prev_keypoints, None
        )
        
        if curr_keypoints is None:
            self.prev_frame = curr_gray
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix
            
        valid_curr = curr_keypoints[status.ravel() == 1]
        valid_prev = self.prev_keypoints[status.ravel() == 1]
        
        if len(valid_curr) > 10:
            m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
            if m is not None:
                self.warp_matrix = m.astype(np.float32)
            else:
                self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.prev_frame = curr_gray
        self.prev_keypoints = valid_curr.reshape(-1, 1, 2) if len(valid_curr) > 0 else None
        
        return self.warp_matrix

    def apply_to_bbox(self, bbox, warp_matrix=None):
        if warp_matrix is None:
            warp_matrix = self.warp_matrix
        
        corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        transformed = cv2.transform(corners, warp_matrix)
        transformed = transformed.reshape(-1, 2)
        
        return [
            int(transformed[0, 0]),
            int(transformed[0, 1]),
            int(transformed[1, 0]),
            int(transformed[1, 1])
        ]


# ============================================
# THREAD-SAFE STORAGE
# ============================================
class ThreadSafeTracklets:
    def __init__(self, max_size=1000):
        self._data = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def __contains__(self, key):
        with self._lock: return key in self._data
    
    def __getitem__(self, key):
        with self._lock:
            self._data.move_to_end(key)
            return self._data[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            if key in self._data: self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
    
    def remove_expired(self, curr_time, timeout):
        with self._lock:
            expired = []
            for k, v in self._data.items():
                if curr_time - v.get("last_seen", 0) > timeout:
                    expired.append(k)
            for k in expired: del self._data[k]
            return expired


# --- SETTINGS ---
CAMERA_ID = "cam_01_ist"
VIDEO_SOURCE = "./videolar/videom5.mp4"
#VIDEO_SOURCE = 0 
GROUP_ID = "istanbul_avm"
HOMOGRAPHY_PATH = "h_cam_01_ist.npy"
YOLO_MODEL_PATH = './models/yolov8x-worldv2.engine' 
REID_ENGINE_PATH = './models/Dino/dino_vitl16_fp16.engine'    
TRACKER_CONFIG = 'bytetrack.yaml'

# Redis Configuration
STREAM_NAME = "track_events"  # MUST match central_tracker_service.py

CONF_THRES_DETECTION = 0.1   
CONF_THRES_HIGH = 0.5        
CONF_THRES_LOW = 0.1         
MIN_QUALITY_THRESHOLD = 0.45 
REID_UPDATE_INTERVAL = 1.0
GP_UPDATE_INTERVAL = 5       
TRACK_LOST_TIMEOUT = 3.0

COLOR_PENDING = (0, 0, 255)     
COLOR_TRACKED = (0, 255, 0)     
COLOR_STAFF   = (255, 0, 255)   
COLOR_TEXT    = (255, 255, 255)

latest_tracks_from_central = [] 
tracks_lock = threading.Lock()
edge_tracklets = ThreadSafeTracklets(max_size=500)

# --- REDIS SETUP ---
try:
    # r_json used for Streams (text-based keys/values)
    r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # r_bytes used for Image Publishing
    r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    
    r_json.ping()
except Exception as e:
    print(f"Redis Connection Error: {e}")
    exit()


def calculate_quality_score(frame, bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    w = x2 - x1; h = y2 - y1
    geo_score = 1.0
    margin = 10 
    
    if x1 < margin or y1 < margin or x2 > frame_width - margin or y2 > frame_height - margin: 
        geo_score -= 0.4
    if w > 0:
        aspect = h / float(w)
        if aspect < 0.4: return 0.0 
        elif aspect < 1.2: geo_score -= 0.15 
        elif aspect > 3.5: geo_score -= 0.15 
    if (w * h) < (60 * 120): geo_score -= 0.2

    blur_score = 0.0
    try:
        y1_c, y2_c = max(0, y1), min(frame_height, y2)
        x1_c, x2_c = max(0, x1), min(frame_width, x2)
        if x2_c > x1_c and y2_c > y1_c:
            crop = frame[y1_c:y2_c, x1_c:x2_c]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_val / 300.0)
    except Exception: pass
    
    final_score = (0.6 * blur_score) + (0.4 * max(0.0, geo_score))
    return max(0.0, final_score)


def redis_listener():
    """
    Listens for Visualization updates from Central.
    Central sends these via Pub/Sub (even if input is Stream).
    """
    global latest_tracks_from_central
    # Use a separate client for blocking PubSub listener to avoid conflicts
    r_sub = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    pubsub = r_sub.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(f"results_viz_stream:{CAMERA_ID}")
    
    print(f"[{CAMERA_ID}] Listening for Viz updates on channel: results_viz_stream:{CAMERA_ID}")
    
    for message in pubsub.listen():
        try:
            data = json.loads(message['data'])
            with tracks_lock:
                latest_tracks_from_central = data
        except Exception as e: 
            print(f"Viz Listener Error: {e}")


def send_track_event(event_type, edge_id, gp=None, bbox=None, feat=None, conf=None, quality=0.0):
    """
    FIXED: Sends event to Redis Stream instead of Pub/Sub.
    """
    packet = {
        "camera_id": CAMERA_ID, 
        "group_id": GROUP_ID, 
        "timestamp": time.time(),
        "event_type": event_type, 
        "edge_track_id": int(edge_id),
        "gp_coord": gp.tolist() if isinstance(gp, np.ndarray) else gp,
        "bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else (list(bbox) if bbox else None),
        "feature": feat.tolist() if isinstance(feat, np.ndarray) else feat,
        "conf": float(conf) if conf is not None else None, 
        "frame_res": [frame_w, frame_h],
        "quality": float(quality)
    }
    
    try:
        # P1 FIX: Use XADD for Redis Streams
        # The central tracker expects a field named 'data' containing the JSON string
        r_json.xadd(STREAM_NAME, {'data': json.dumps(packet)}, maxlen=100000)
    except Exception as e:
        print(f"Redis Stream Error: {e}")


# --- MAIN ---
print(f"[{CAMERA_ID}] Loading YOLO & DINOv3 Models...")
model = YOLO(YOLO_MODEL_PATH)
reid_extractor = TensorRTReidExtractor(REID_ENGINE_PATH)
cmc = CameraMotionCompensation()

try: 
    H = np.load(HOMOGRAPHY_PATH)
    print(f"[{CAMERA_ID}] Homography loaded.")
except: 
    H = None
    print(f"[{CAMERA_ID}] No homography found, using pixel coordinates.")

cap = cv2.VideoCapture(VIDEO_SOURCE)
threading.Thread(target=redis_listener, daemon=True).start()

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FRAME_COUNT = 0

print(f"[{CAMERA_ID}] Started. Resolution: {frame_w}x{frame_h}. Streaming to Redis STREAM: {STREAM_NAME}...")

while True:
    ret, frame = cap.read()
    if not ret: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        
    FRAME_COUNT += 1
    curr_time = time.time()
    
    warp_matrix = cmc.compute(frame)
    annotated = frame.copy()
    
    # Get central tracker results (thread-safe)
    with tracks_lock:
        central_map = {t['edge_track_id']: t for t in latest_tracks_from_central}
    
    # Run YOLO tracker
    results = model.track(
        frame, 
        stream=False, 
        verbose=False, 
        classes=[0], 
        persist=True, 
        tracker=TRACKER_CONFIG, 
        conf=CONF_THRES_DETECTION, 
        iou=0.8
    )
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for bbox, conf, eid in zip(boxes, confs, ids):
            x1, y1, x2, y2 = bbox
            box_color = COLOR_PENDING
            top_label = ""
            
            # Check central tracker results
            if eid in central_map:
                c_data = central_map[eid]
                if c_data.get('is_staff', False):
                    box_color = COLOR_STAFF
                    name = c_data.get('name', 'Staff')
                    top_label = f"{name}"
                else:
                    box_color = COLOR_TRACKED
                    gid = c_data.get('global_id', '?')
                    top_label = f"G {gid}"
            
            if conf < CONF_THRES_HIGH:
                box_color = (100, 100, 100)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            if top_label:
                (tw, th), _ = cv2.getTextSize(top_label, 0, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(annotated, top_label, (x1 + 5, y1 - 5), 0, 0.7, COLOR_TEXT, 2)
            
            # Compute ground plane coordinates
            if H is not None:
                p = np.array([[(x1+x2)/2, y2]], dtype='float32').reshape(-1, 1, 2)
                gp_raw = cv2.perspectiveTransform(p, H)[0][0]
                gp = np.array([gp_raw[0], gp_raw[1]])
            else:
                gp = np.array([float((x1+x2)/2), float((y1+y2)/2)])
            
            # Initialize new tracklet
            if eid not in edge_tracklets:
                edge_tracklets[eid] = {
                    "best_q": 0.0, 
                    "last_reid": 0.0, 
                    "last_gp": 0, 
                    "last_seen": curr_time,
                    "last_bbox": list(bbox), 
                    "predicted_bbox": None
                }
                send_track_event("TRACK_NEW", eid, gp, bbox, conf=conf)

            t_data = edge_tracklets[eid]
            t_data["last_seen"] = curr_time
            t_data["last_bbox"] = list(bbox)
            
            # Smart Sparse Re-ID Logic
            if conf >= CONF_THRES_HIGH:
                q_score = calculate_quality_score(frame, bbox, frame_w, frame_h)
                do_reid = False
                
                if q_score > MIN_QUALITY_THRESHOLD:
                    if t_data["best_q"] == 0.0:
                        do_reid = True
                    elif q_score > (t_data["best_q"] * 1.2):
                        do_reid = True
                    elif (curr_time - t_data["last_reid"]) > REID_UPDATE_INTERVAL:
                        do_reid = True
                
                if do_reid:
                    feats, _ = reid_extractor.extract_features(frame, [bbox])
                    if feats.size > 0:
                        t_data["best_q"] = max(t_data["best_q"], q_score)
                        t_data["last_reid"] = curr_time
                        send_track_event(
                            "TRACK_UPDATE_FEATURE", eid, gp, bbox, 
                            feat=feats[0], conf=conf, quality=q_score
                        )
                        cv2.circle(annotated, (x2-5, y1+5), 3, (0, 255, 255), -1)
            
            elif conf >= CONF_THRES_LOW:
                if (FRAME_COUNT - t_data["last_gp"]) >= GP_UPDATE_INTERVAL:
                    send_track_event("TRACK_UPDATE_GP", eid, gp, bbox, conf=conf)
                    t_data["last_gp"] = FRAME_COUNT

    # Thread-safe lost track handling
    lost_keys = edge_tracklets.remove_expired(curr_time, TRACK_LOST_TIMEOUT)
    for k in lost_keys:
        send_track_event("TRACK_LOST", k)

    # Publish annotated frame
    ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if ret:
        r_bytes.set(f"live_feed:{CAMERA_ID}", buffer.tobytes(), ex=5)

cap.release()