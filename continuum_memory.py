import numpy as np
from numpy.linalg import norm
import json
import time

class ContinuumState:
    """
    Refactored Nested Learning Module with Probabilistic (Bayesian) Memory.
    Includes Soft Gating, Breakout, Attention Buffer, and Variance Estimation.
    """
    def __init__(self, feature_dim=1024, data=None):
        # --- Hyperparameters ---
        self.ALPHA_SLOW = 0.05   
        self.STABILITY_THRESH = 0.65 
        self.SIGMOID_SCALE = 10.0   
        self.BREAKOUT_LIMIT = 30    
        self.BOOTSTRAP_FRAMES = 15
        self.BUFFER_SIZE = 5

        if data:
            self.fast_buffer = [np.array(v, dtype=np.float32) for v in data.get('fast_buffer', [])]
            self.slow = np.array(data['slow'], dtype=np.float32)
            # --- NEW: Load Variance ---
            self.slow_var = np.array(data.get('slow_var'), dtype=np.float32) if 'slow_var' in data else np.zeros(feature_dim, dtype=np.float32)
            
            self.count = data.get('count', 0)
            self.last_update = data.get('last_update', time.time())
            self.divergence_counter = data.get('divergence_counter', 0)
        else:
            self.fast_buffer = []
            self.slow = np.zeros(feature_dim, dtype=np.float32)
            self.slow_var = np.zeros(feature_dim, dtype=np.float32) # Initialize Variance
            self.count = 0
            self.last_update = time.time()
            self.divergence_counter = 0

    def _normalize(self, v):
        """L2 Normalization."""
        return v / (norm(v) + 1e-6)

    def learn(self, vector, quality=1.0):
        """
        Optimization Step with Variance Estimation.
        """
        feat = self._normalize(np.array(vector, dtype=np.float32))
        
        # Initialization
        if self.count == 0:
            self.fast_buffer = [feat]
            self.slow = feat
            self.slow_var = np.ones_like(feat) * 0.01 # Initial high uncertainty
            self.count = 1
            self.last_update = time.time()
            return

        # 1. Update Fast Buffer
        self.fast_buffer.append(feat)
        if len(self.fast_buffer) > self.BUFFER_SIZE:
            self.fast_buffer.pop(0)

        # Calculate Centroid (Input for Slow Memory)
        fast_centroid = np.mean(self.fast_buffer, axis=0)
        fast_centroid = self._normalize(fast_centroid)
        
        # 2. Consistency Check
        consistency = np.dot(fast_centroid, self.slow)
        
        # Sigmoid Modulation
        modulation = 1.0 / (1.0 + np.exp(-self.SIGMOID_SCALE * (consistency - self.STABILITY_THRESH)))
        
        # Bootstrap Override
        if self.count < self.BOOTSTRAP_FRAMES:
            modulation = 1.0
            current_alpha_slow = (self.ALPHA_SLOW * 1.5) * modulation
        else:
            current_alpha_slow = self.ALPHA_SLOW * modulation

        # 3. Breakout Mechanism
        if consistency < self.STABILITY_THRESH and self.count >= self.BOOTSTRAP_FRAMES:
            self.divergence_counter += 1
            if self.divergence_counter > self.BREAKOUT_LIMIT:
                # Snap memory (Mean & Var)
                self.slow = (0.5 * self.slow) + (0.5 * fast_centroid)
                self.slow = self._normalize(self.slow)
                self.slow_var = np.ones_like(self.slow) * 0.05 # Reset variance
                self.divergence_counter = 0 
        else:
            self.divergence_counter = max(0, self.divergence_counter - 1)

        # 4. Update Slow Memory (Mean AND Variance)
        # Calculate difference between new input and current memory
        diff = fast_centroid - self.slow
        
        # Update Mean
        self.slow = (1 - current_alpha_slow) * self.slow + (current_alpha_slow * fast_centroid)
        self.slow = self._normalize(self.slow)
        
        # --- NEW: Update Variance (Running estimation) ---
        # Var_t = (1-a)*Var_{t-1} + a*(diff^2)
        # We square the difference to get magnitude of variation
        sq_diff = diff ** 2
        self.slow_var = (1 - current_alpha_slow) * self.slow_var + (current_alpha_slow * sq_diff)

        self.count += 1
        self.last_update = time.time()

    def get_identity(self):
        """Returns (Mean, Variance) tuple."""
        return (self.slow, self.slow_var) if self.count > 0 else None

    def to_dict(self):
        return {
            "fast_buffer": [v.tolist() for v in self.fast_buffer],
            "slow": self.slow.tolist(),
            "slow_var": self.slow_var.tolist(), # Serialize variance
            "count": self.count,
            "last_update": self.last_update,
            "divergence_counter": self.divergence_counter
        }