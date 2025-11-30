import os
import cv2
import numpy as np
import torch
import configparser
from tqdm import tqdm

# --- AYARLAR ---
# BURAYI DÜZENLE: Elindeki dosyanın tam yolu (Windows kullanıyorsan başına r koy)
WEIGHTS_PATH = r"C:\Users\metev\OneDrive\Desktop\CmptVsn\Pose Olmadan Son Model\SonProje-v7-Dino\models\Dino\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

def load_local_dinov3():
    print(f"[Model] Loading Local DINOv3 from: {WEIGHTS_PATH}")
    
    # 1. Model İskeletini Oluştur (Ağırlıksız - Pretrained=False)
    # Hata aldığın için 'source=local' kullanmayı deneyebiliriz ama önce github deniyoruz.
    # Önceki hatanda repo zaten inmişti, sadece ağırlık inmemişti.
    try:
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', pretrained=False)
    except:
        # Eğer Hub hatası verirse, yerel cache yolunu dene (Önceki hatandan aldığım yol)
        # Windows kullanıcısı olduğun için yol şuna benzer olabilir:
        repo_dir = os.path.expanduser(r'~/.cache/torch/hub/facebookresearch_dinov3_main')
        if os.path.exists(repo_dir):
             model = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', pretrained=False)
        else:
             print("HATA: DINOv3 repo'su bulunamadı. Lütfen internetten bir kez indirmeyi deneyin.")
             raise

    # 2. Yerel Ağırlıkları Yükle
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {WEIGHTS_PATH}")
        
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

    # 3. Checkpoint İçinden Ağırlıkları Ayıkla
    # DINO checkpointleri genelde {'model': ...} veya {'teacher': ...} yapısındadır.
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint

    # 4. Prefix Temizliği (Gerekirse)
    # Bazen "module." veya "backbone." öneki olur, bunları temizlemeliyiz.
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[k] = v

    # 5. Ağırlıkları Modele Enjekte Et
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Model] Ağırlıklar Yüklendi. Eksik anahtarlar: {len(msg.missing_keys)}")
    
    return model.cuda().eval()

# --- MODELİ YÜKLE ---
model = load_local_dinov3()

DATA_ROOT = "models/Dino/MOT17/train"
OUTPUT_DIR = "models/processed_mot17_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(img_crop):
    """
    DINOv3 (ViT-L/16) için ön işleme.
    Patch size 16 olduğu için giriş boyutu 16'nın katı olmalı (224x224).
    """
    img = cv2.resize(img_crop, (224, 224)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    # ImageNet Mean/Std
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    img = img.transpose(2, 0, 1)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()

def process_sequence(seq_name):
    seq_path = os.path.join(DATA_ROOT, seq_name)
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    img_dir = os.path.join(seq_path, 'img1')
    
    if not os.path.exists(gt_path):
        print(f"Skipping {seq_name} (GT not found)")
        return

    gt_data = np.loadtxt(gt_path, delimiter=',')
    # Class: 1=Pedestrian, 2=Person on Vehicle, 7=Static Person
    gt_data = gt_data[np.isin(gt_data[:, 7], [1, 2, 7])]
    
    frames = np.unique(gt_data[:, 0])
    sequence_data = []
    
    print(f"Processing {seq_name}...")
    for frame_idx in tqdm(frames):
        img_name = f"{int(frame_idx):06d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
        
        full_img = cv2.imread(img_path)
        frame_rows = gt_data[gt_data[:, 0] == frame_idx]
        
        frame_features = []
        frame_meta = []
        
        h_img, w_img, _ = full_img.shape
        
        for row in frame_rows:
            fid, pid, x, y, w, h = row[:6]
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x+w), min(h_img, y+h)
            
            if (x2-x1) < 10 or (y2-y1) < 10: continue
            
            crop = full_img[y1:y2, x1:x2]
            
            # Feature Extraction
            with torch.no_grad():
                tensor_img = preprocess_image(crop)
                emb = model(tensor_img).cpu().numpy()[0]
                emb = emb / (np.linalg.norm(emb) + 1e-6)
            
            geo = [x1/w_img, y1/h_img, w/w_img, h/h_img]
            
            frame_features.append(emb)
            frame_meta.append({'id': int(pid), 'geo': geo})
            
        if frame_features:
            sequence_data.append({
                'frame': int(frame_idx),
                'features': np.array(frame_features, dtype=np.float32),
                'meta': frame_meta
            })
            
    np.save(os.path.join(OUTPUT_DIR, f"{seq_name}.npy"), sequence_data, allow_pickle=True)

# Main Loop
if os.path.exists(DATA_ROOT):
    seqs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    for s in seqs:
        process_sequence(s)
    print("Veri Hazırlığı Tamamlandı! Eğitim için hazırsın.")
else:
    print(f"HATA: {DATA_ROOT} bulunamadı. Lütfen MOT17 klasör yolunu kontrol et.")