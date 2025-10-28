## FEATURE ENGINEERING ##

import numpy as np
from scipy import ndimage

def interface_count(grid, a, b):
    # 4-neighbor interfaces between labels a and b
    g = grid
    right = (g[:, :-1] == a) & (g[:, 1:] == b)
    right |= (g[:, :-1] == b) & (g[:, 1:] == a)
    down = (g[:-1, :] == a) & (g[1:, :] == b)
    down |= (g[:-1, :] == b) & (g[1:, :] == a)
    return right.sum() + down.sum()

def connectivity_features(grid, label):
    mask = (grid == label).astype(np.uint8)
    if mask.sum() == 0:
        return 0, 0, 0.0
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)  # 4-neighborhood
    comp, n = ndimage.label(mask, structure=struct)
    sizes = np.bincount(comp.ravel())[1:]
    return n, sizes.max(), sizes.mean()

def symmetry_scores(grid):
    h = 1.0 - (np.mean(grid == np.fliplr(grid)))
    v = 1.0 - (np.mean(grid == np.flipud(grid)))
    return h, v

def cooccurrence_4n(grid, n_labels=5):
    # counts of adjacent pairs (unordered) across 4-neighbors
    counts = np.zeros((n_labels, n_labels), dtype=np.int32)
    # right neighbors
    a, b = grid[:, :-1], grid[:, 1:]
    for i in range(n_labels):
        for j in range(i, n_labels):
            c = ((a == i) & (b == j)) | ((a == j) & (b == i))
            counts[i, j] += c.sum()
            if i != j:
                counts[j, i] = counts[i, j]
    # down neighbors
    a, b = grid[:-1, :], grid[1:, :]
    for i in range(n_labels):
        for j in range(i, n_labels):
            c = ((a == i) & (b == j)) | ((a == j) & (b == i))
            counts[i, j] += c.sum()
            if i != j:
                counts[j, i] = counts[i, j]
    return counts

def calc_avg_distances(grid, label_a, label_b):
    """Calculate sum of Manhattan distances between all cells of label_a and all cells of label_b"""
    cells_a = np.argwhere(grid == label_a)
    cells_b = np.argwhere(grid == label_b)
    
    if len(cells_a) == 0 or len(cells_b) == 0:
        return 0
    
    total_dist = 0
    for pos_a in cells_a:
        for pos_b in cells_b:
            total_dist += np.abs(pos_a[0] - pos_b[0]) + np.abs(pos_a[1] - pos_b[1])
    
    return total_dist / len(cells_a)

def calc_max_distances(grid, label_a, label_b):
    """Calculate sum of Manhattan distances between all cells of label_a and all cells of label_b"""
    cells_a = np.argwhere(grid == label_a)
    cells_b = np.argwhere(grid == label_b)
    
    if len(cells_a) == 0 or len(cells_b) == 0:
        return 0
    
    max_dist = 0
    for pos_a in cells_a:
        for pos_b in cells_b:
            dist = np.abs(pos_a[0] - pos_b[0]) + np.abs(pos_a[1] - pos_b[1])
            max_dist = dist if dist>max_dist else max_dist
    
    return max_dist


def feature_eng_train(grids):
    # grids: (N, 7, 7) int
    N = grids.shape[0]
    feats = []
    for g in grids:
        # Annotate feature IDs for engineered features (starting from feature id 49)
        # After the flattened grid (feature ids 0-48), the engineered features are:
        # 49: counts[0] (label 0 cell count)
        # 50: counts[1] (label 1 cell count)
        # 51: counts[2] (label 2 cell count)
        # 52: counts[3] (label 3 cell count)
        # 53: counts[4] (label 4 cell count)
        # 54: props[0] (label 0 proportion)
        # 55: props[1] (label 1 proportion)
        # 56: props[2] (label 2 proportion)
        # 57: props[3] (label 3 proportion)
        # 58: props[4] (label 4 proportion)
        # 59: total_iface
        # 60-69: ifaces (interface_count for each label pair, 10 pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        # 70-74: ns (number of connected components per label 0-4)
        # 75-79: maxs (max component size per label 0-4)
        # 80-84: means (mean component size per label 0-4)
#        # 85: sym_h (horizontal symmetry score)
#        # 86: sym_v (vertical symmetry score)
        # 87-96: avg_distances (pairwise avg Manhattan distances, label pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        # 97-106: centroid_pair_l1 (pairwise centroid L1 distances, same label pairs as above)
        # 107-116: centroid_pair_l2 (pairwise centroid L2 distances, same label pairs as above)
        # 117-121: centroid_to_center_l1 (centroid-to-center L1, per label 0-4)
        # 122-126: centroid_to_center_l2 (centroid-to-center L2, per label 0-4)
        # 127-131: intra_pair_mean_l1 (mean intra-label pairwise L1, per label 0-4)
        # 132-136: l1_radius (mean absolute deviation to centroid, per label 0-4)
        # 137-146: max_distances (pairwise max Manhattan distances, label pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        # 147-156: pair_min_l1 (pairwise min cityblock distances, label pairs as above)
        # 157-166: pair_hausdorff_l1 (pairwise Hausdorff cityblock distances, label pairs as above)
        # 167-171: intra_pair_max_l1 (max intra-label pairwise L1, per label 0-4)
        # 172-176: min_border (min distance to border, per label 0-4)
        # 177-181: mean_border (mean distance to border, per label 0-4)
        # 182-186: center_counts (center 3x3 cell counts per label 0-4)
        # 187-191: border_counts (border cell counts per label 0-4)
        # 192-201: prop_pairs (pairwise label proportions, pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
#        # 202-206: largest_ratios (max component size / count, per label 0-4)
        # 207-211: frags (num components / count, per label 0-4)
        # 212-216: center_ratios (center / (center+border), per label 0-4)
        # 217: iface_density (total_iface / 84.0)
        # 218-227: iface_norm (interface_count / (count_i + count_j), pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
#        # 228-232: sym_h_props (sym_h * prop, per label 0-4)
#        # 233-237: sym_v_props (sym_v * prop, per label 0-4)
        # The order matches the feat_vec construction above.
        # For exact mapping, see the feat_vec definition in the code.
        counts = [(g == k).sum() for k in range(5)]
        props = [c / 49.0 for c in counts]

        # distances
        avg_distances = [calc_avg_distances(g, i, j) for i in range(5) for j in range(i+1,5)]
        max_distances = [calc_max_distances(g, i, j) for i in range(5) for j in range(i+1,5)]

        # precompute coordinates and centroids
        coords = [np.argwhere(g == k) for k in range(5)]
        h, w = g.shape
        center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float32)
        centroids = [
            c.mean(axis=0).astype(np.float32) if len(c) else np.array([0.0, 0.0], dtype=np.float32)
            for c in coords
        ]

        # pairwise centroid distances (L1, L2)
        centroid_pair_l1, centroid_pair_l2 = [], []
        for i in range(5):
            for j in range(i + 1, 5):
                if len(coords[i]) == 0 or len(coords[j]) == 0:
                    d1, d2 = 0.0, 0.0
                else:
                    diff = centroids[i] - centroids[j]
                    d1 = float(np.abs(diff).sum())
                    d2 = float(np.sqrt((diff ** 2).sum()))
                centroid_pair_l1.append(d1)
                centroid_pair_l2.append(d2)

        # centroid to grid-center distances (per label, L1, L2)
        centroid_to_center_l1, centroid_to_center_l2 = [], []
        for k in range(5):
            if len(coords[k]) == 0:
                centroid_to_center_l1.append(0.0)
                centroid_to_center_l2.append(0.0)
            else:
                diff = centroids[k] - center
                centroid_to_center_l1.append(float(np.abs(diff).sum()))
                centroid_to_center_l2.append(float(np.sqrt((diff ** 2).sum())))

        # intra-label dispersion (mean/max pairwise L1) and L1 radius (MAD to centroid)
        intra_pair_mean_l1, intra_pair_max_l1, l1_radius = [], [], []
        for k in range(5):
            c = coords[k]
            if len(c) <= 1:
                intra_pair_mean_l1.append(0.0)
                intra_pair_max_l1.append(0.0)
                l1_radius.append(0.0)
            else:
                diffs = np.abs(c[:, None, :] - c[None, :, :]).sum(axis=2)
                n = diffs.shape[0]
                tri = diffs[np.triu_indices(n, 1)]
                intra_pair_mean_l1.append(float(tri.mean()))
                intra_pair_max_l1.append(float(tri.max()))
                l1_radius.append(float(np.abs(c - centroids[k]).sum(axis=1).mean()))

        # distance to border per label (min, mean)
        min_border, mean_border = [], []
        for k in range(5):
            c = coords[k]
            if len(c) == 0:
                min_border.append(0.0)
                mean_border.append(0.0)
            else:
                r = c[:, 0]
                ccol = c[:, 1]
                d = np.minimum.reduce([r, ccol, (h - 1) - r, (w - 1) - ccol]).astype(np.float32)
                min_border.append(float(d.min()))
                mean_border.append(float(d.mean()))

        # pairwise min and Hausdorff (cityblock) distances via distance transform
        dt = [None] * 5
        for k in range(5):
            if len(coords[k]):
                dt[k] = ndimage.distance_transform_cdt(g != k, metric='taxicab').astype(np.float32)

        pair_min_l1, pair_hausdorff_l1 = [], []
        for i in range(5):
            for j in range(i + 1, 5):
                if len(coords[i]) == 0 or len(coords[j]) == 0:
                    pair_min_l1.append(0.0)
                    pair_hausdorff_l1.append(0.0)
                else:
                    d_ij_min = float(dt[i][g == j].min())
                    d_ji_min = float(dt[j][g == i].min())
                    pair_min_l1.append(min(d_ij_min, d_ji_min))

                    h_ij = float(dt[j][g == i].max())
                    h_ji = float(dt[i][g == j].max())
                    pair_hausdorff_l1.append(max(h_ij, h_ji))

        # append new features
        avg_distances.extend(centroid_pair_l1)
        avg_distances.extend(centroid_pair_l2)
        avg_distances.extend(centroid_to_center_l1)
        avg_distances.extend(centroid_to_center_l2)
        avg_distances.extend(intra_pair_mean_l1)
        avg_distances.extend(l1_radius)

        max_distances.extend(pair_min_l1)
        max_distances.extend(pair_hausdorff_l1)
        max_distances.extend(intra_pair_max_l1)
        max_distances.extend(min_border)
        max_distances.extend(mean_border)

        # interfaces
        iface = {(i, j): interface_count(g, i, j) for i in range(5) for j in range(i+1, 5)}
        ifaces = (interface_count(g, i, j) for i in range(5) for j in range(i+1, 5))
        total_iface = sum(ifaces)

        # connectivity
        n0, max0, mean0 = connectivity_features(g, 0)
        n1, max1, mean1 = connectivity_features(g, 1)
        n2, max2, mean2 = connectivity_features(g, 2)
        n3, max3, mean3 = connectivity_features(g, 3)
        n4, max4, mean4 = connectivity_features(g, 4)
        ns = [n0, n1, n2, n3, n4]
        maxs = [max0, max1, max2, max3, max4]
        means = [mean0, mean1, mean2, mean3, mean4]

        # symmetry
        sym_h, sym_v = symmetry_scores(g)

        # center vs border for label 0
        center_mask = np.zeros_like(g, dtype=bool)
        center_mask[2:5, 2:5] = True
        # Center vs border for all labels (0 through 4)
        center_counts = [(g[center_mask] == k).sum() for k in range(5)]
        border_counts = [(g[~center_mask] == k).sum() for k in range(5)]

        # interactions
        prop_pairs = [props[i]*props[j] for i in range(5) for j in range(i+1, 5)]
        sdiv = lambda a,b: a / (b + 1e-6)
        largest_ratios = [sdiv(maxs[k], counts[k]) for k in range(5)]

        frags = [sdiv(ns[k], counts[k]) for k in range(5)]
        center_ratios = [sdiv(center_counts[k], center_counts[k] + border_counts[k]) for k in range(5)]
        iface_density = sdiv(total_iface, 84.0)  # total 4-neighbor edges in 7x7
        iface_norm = (sdiv(iface[(i,j)], counts[i] + counts[j]) for i in range(5) for j in range(i+1,5))
        # Symmetry-proportion interactions for all labels (0 through 4)
        sym_h_props = [sym_h * props[k] for k in range(5)]
        sym_v_props = [sym_v * props[k] for k in range(5)]

        # Build feature vector
        feat_vec = [
            *counts, *props,
            total_iface, *ifaces, 
            *ns, *maxs, *means,
            sym_h, sym_v,
            *avg_distances, *max_distances,
            *center_counts, *border_counts,
            *prop_pairs,
            *largest_ratios, *frags, *center_ratios,
            iface_density, *iface_norm,
            *sym_h_props, *sym_v_props
        ]
        feats.append(feat_vec)

    features = np.asarray(feats, dtype=np.float32)
    grids_flat = grids.reshape(-1, 49) # flatten the grids
    grids_out = np.hstack([grids_flat, features]) #stack the features horizontally with the flattened grids
    return grids_out.astype(np.float32)

def append_district_counts(grids): #performs the feature engineering to add district counts
    grids_flat = grids.reshape(-1, 49) #first flatten the grids

    counts = [np.sum(grids_flat==0, axis=1),
              np.sum(grids_flat==1, axis=1),
              np.sum(grids_flat==2, axis=1),
              np.sum(grids_flat==3, axis=1),
              np.sum(grids_flat==4, axis=1)] #list of 5 length n_grids arrays containing counts of each district
    features = np.stack(counts).T #stack and transpose counts to get n_grids x 5 array
    grids_out = np.hstack([grids_flat, features]) #stack the features horizontally with the flattened grids
    return grids_out.astype(np.float32)
