import torch
import torch.nn.functional as F

def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    GPU-Optimized k-reciprocal Encoding Re-Ranking.
    
    Args:
        probFea (numpy.ndarray or torch.Tensor): Query features (M, dim)
        galFea (numpy.ndarray or torch.Tensor): Gallery features (N, dim)
        k1 (int): k-reciprocal nearest neighbors
        k2 (int): k-nearest neighbors for query expansion
        lambda_value (float): Weight for the original distance
        
    Returns:
        final_dist (numpy.ndarray): Re-ranked distance matrix (M, N)
    """
    # 1. Setup & Device Management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure inputs are tensors on the correct device
    if not isinstance(probFea, torch.Tensor):
        query = torch.from_numpy(probFea).to(device)
    else:
        query = probFea.to(device)
        
    if not isinstance(galFea, torch.Tensor):
        gallery = torch.from_numpy(galFea).to(device)
    else:
        gallery = galFea.to(device)

    # FP16 optimization for speed/memory (Optional, safe for Re-ID)
    query = query.half() if query.is_floating_point() else query
    gallery = gallery.half() if gallery.is_floating_point() else gallery

    num_query = query.shape[0]
    num_gallery = gallery.shape[0]
    all_num = num_query + num_gallery
    
    # Concatenate features
    features = torch.cat([query, gallery], dim=0)

    # 2. Optimized Euclidean Distance (torch.cdist is C++ optimized)
    # (N, d) -> (N, N) distance matrix
    original_dist = torch.cdist(features, features).float() # Convert back to float32 for precision in ranking
    
    # Power trick to normalize distance to [0, 1] roughly for the kernel
    original_dist = original_dist / (torch.max(original_dist, dim=0)[0] + 1e-6)
    
    # Keep the query-gallery block for the final result
    original_dist_qg = original_dist[:num_query, num_query:]

    # 3. Initial Ranking (Top-K)
    # We only need the top (k1 + 1) neighbors for the check
    # taking more to be safe for reciprocity check
    _, initial_rank = torch.topk(original_dist, k=k1+50, dim=1, largest=False)
    
    # 4. K-Reciprocal Neighbor Search (Vectorized/Semi-Vectorized)
    # Creating a dense mask is too heavy (N*N), so we iterate efficiently
    
    # Transpose rank for easier indexing
    initial_rank = initial_rank.long()
    
    # Weighted Jaccard Matrix (V)
    # We use a sparse approach logic but with dense tensors for small N (<20k)
    # Or strict neighbor list processing
    
    # -- Fast Setup of V Matrix --
    # Initialize all self-dist as 0 (perfect match)
    
    # Note: Full vectorization of the Expansion Step is complex. 
    # This implementation focuses on vectorizing the Jaccard distance 
    # which is the O(N^2) bottleneck.
    
    nn_k1 = initial_rank[:, :k1] # (N, k1)
    
    # Check reciprocity using broadcasting (Heavy memory usage for large N, handle with care)
    # For optimized path, we iterate but purely on GPU tensors
    
    # Prepare sparse components
    rows = []
    cols = []
    vals = []
    
    # Using a simplified batch loop for k-reciprocal to keep memory low
    # This loop is N times but very fast on GPU ops
    for i in range(all_num):
        # Forward neighbors
        f_idx = nn_k1[i] # (k1,)
        
        # Backward neighbors of the forward neighbors
        # We need to see if 'i' is in the neighborhood of 'f_idx'
        # batch lookup: (k1, k1)
        b_idx = nn_k1[f_idx] 
        
        # Check masks: where does 'i' appear in b_idx?
        reciprocal_mask = (b_idx == i).any(dim=1) # (k1,)
        
        # Get valid reciprocal indices
        k_reciprocal_idx = f_idx[reciprocal_mask]
        
        # Expansion (Candidates of candidates)
        # If we want strict implementation of k/2 expansion:
        if k_reciprocal_idx.shape[0] > 0:
             # Just use the simple expansion: neighbors of reciprocal
            candidates = nn_k1[k_reciprocal_idx] # (R, k1)
            candidates_flat = candidates.view(-1)
            
            # Simple unique via mask (or just include all and let Jaccard handle weights)
            # To strictly follow paper, we verify reciprocity of candidates, but for speed
            # we often skip or do a simplified check.
            # Here we just add the k_reciprocal_idx themselves as strong connections.
            
            # Weight calculation
            current_dist = original_dist[i, k_reciprocal_idx]
            current_weights = torch.exp(-current_dist)
            
            rows.append(torch.full_like(k_reciprocal_idx, i))
            cols.append(k_reciprocal_idx)
            vals.append(current_weights)

    # Stack sparse data
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)
    
    # Create Sparse Matrix V (N, N)
    V = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), 
        vals, 
        (all_num, all_num)
    ).to_dense() # Convert to dense for fast Jaccard (if N < 20000, this fits in VRAM)

    # 5. Query Expansion (k2)
    if k2 > 1:
        # Average the V vectors of the k2 nearest neighbors
        # V_qe = (V[top_k2].sum(1)) / k2
        # Use slicing
        top_k2_indices = initial_rank[:, :k2]
        # Gather V rows: (N, k2, N) -> This is huge.
        # Optimized: V_qe[i] = mean(V[neighbors[i]])
        # We can use matrix multiplication: V_qe = T * V
        # Where T is a normalized adjacency matrix of top-k2 neighbors
        
        # Create T matrix (Transition matrix)
        T_indices = top_k2_indices # (N, k2)
        T_rows = torch.arange(all_num, device=device).unsqueeze(1).expand_as(T_indices)
        T_vals = torch.ones_like(T_indices, dtype=torch.float32) / k2
        
        T = torch.sparse_coo_tensor(
            torch.stack([T_rows.reshape(-1), T_indices.reshape(-1)]),
            T_vals.reshape(-1),
            (all_num, all_num)
        )
        
        # V = T * V (Expansion)
        V = torch.sparse.mm(T, V)

    # 6. Vectorized Jaccard Distance
    # J(A,B) = 1 - (min(A,B) / max(A,B))
    # Using generalized Jaccard for weighted vectors:
    # dist = 1 - sum(min(Vi, Vj)) / sum(max(Vi, Vj))
    
    # To avoid O(N^2) loops, we compute intersection/union in blocks or directly
    # Since V is dense now (N, N):
    
    # We focus only on Query vs Gallery (M, G) to save time
    V_query = V[:num_query]      # (M, N)
    V_gallery = V[num_query:]    # (G, N)
    
    # This part can be heavy. We can use Euclidean on V as a proxy or strict Jaccard.
    # Strict Jaccard on GPU:
    # Intersection = min(V_q.unsqueeze(1), V_g.unsqueeze(0)).sum(-1)
    # This expands to (M, G, N) -> Memory Explosion!
    
    # Optimization: Interaction = V_q @ V_g.T (Dot product is a proxy for Intersection size)
    # For binary Jaccard: |A n B| = A @ B.T
    # For weighted, Dot product is a very good approximation used in "Fast Re-Ranking".
    
    intersection = torch.mm(V_query, V_gallery.t())
    
    # Union area = sum(V_q) + sum(V_g) - intersection (Inclusion-Exclusion Principle)
    # Note: This is valid for binary sets. For weighted, it's an approximation, but fast.
    V_q_sq = (V_query ** 2).sum(1).unsqueeze(1)
    V_g_sq = (V_gallery ** 2).sum(1).unsqueeze(0)
    
    # Refined Jaccard distance formulation
    jaccard_dist = 1.0 - (intersection / (V_q_sq + V_g_sq - intersection + 1e-6))
    
    # 7. Final Fusion
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist_qg * lambda_value
    
    return final_dist.cpu().numpy() # Return numpy to match original API