# topology_manager.py
# Gaussian Smooth Probability Update

import numpy as np
import json
import redis

class TopologyManager:
    def __init__(self, redis_client):
        self.r = redis_client
        self.ALPHA = 0.1 # Learning rate (Yeni verinin etkisi)
        
    def _get_key(self, group_id, src, dst):
        return f"topology:{group_id}:{src}:{dst}"

    def update_topology(self, group_id, cam_src, cam_dst, dt):
        """
        İki kamera arasındaki geçiş süresini (dt) öğrenir.
        Online Mean/Variance update algoritması kullanır.
        """
        if cam_src == cam_dst: return
        key = self._get_key(group_id, cam_src, cam_dst)
        
        data = self.r.get(key)
        if data:
            stats = json.loads(data)
            mu, var, count = stats['mu'], stats['var'], stats['count']
            
            # Online Update (Welford's Algorithm benzeri)
            diff = dt - mu
            incr = self.ALPHA * diff
            new_mu = mu + incr
            new_var = (1 - self.ALPHA) * var + (self.ALPHA * diff * (dt - new_mu))
            
            new_stats = {'mu': new_mu, 'var': new_var, 'count': count + 1}
        else:
            # İlk veri: Varyansı biraz geniş tutalım (Hoşgörü payı)
            new_stats = {'mu': dt, 'var': 25.0, 'count': 1}
            
        self.r.set(key, json.dumps(new_stats))

    def get_transition_prob(self, group_id, cam_src, cam_dst, dt):
        """
        Gaussian Probability Density Function (PDF) kullanarak 
        geçişin ne kadar 'olası' olduğunu 0.0 ile 1.0 arasında döndürür.
        """
        key = self._get_key(group_id, cam_src, cam_dst)
        data = self.r.get(key)
        
        if not data: 
            return 0.5 # Bilinmeyen yol için nötr olasılık
            
        stats = json.loads(data)
        mu = stats['mu']
        sigma = np.sqrt(stats['var']) + 1e-6 # Sıfıra bölünmeyi önle
        
        # Gaussian Formülü: e^(-(x-mu)^2 / (2*sigma^2))
        # Bu bize Çan Eğrisi üzerinde bir nokta verir.
        # Eğer dt tam ortalamadaysa (mu), sonuç 1.0 çıkar.
        # Eğer dt 3 sigma uzaktaysa, sonuç 0.01 civarı çıkar.
        exponent = -((dt - mu)**2) / (2 * (sigma**2))
        prob = np.exp(exponent)
        
        return float(prob)