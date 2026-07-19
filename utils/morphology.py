import numpy as np
import pandas as pd
from scipy import ndimage
import skimage.morphology as morph
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops_table

def refine_mask(binary_mask, min_size=60, closing_disk=6, peak_min_dist=50):
    """
    Instance segmentation pipeline optimized to prevent over-segmentation.
    Converts binary probability masks into distinct quantifiable cellular instances.
    """
    cleaned = morph.remove_small_objects(binary_mask, min_size=min_size)
    cleaned = morph.binary_closing(cleaned, morph.disk(closing_disk))

    if not np.any(cleaned):
        return cleaned, np.zeros_like(cleaned, dtype=bool)

    distance = ndimage.distance_transform_edt(cleaned)
    coords = peak_local_max(distance, min_distance=peak_min_dist, labels=cleaned)

    if len(coords) == 0:
        return cleaned, np.zeros_like(cleaned, dtype=bool)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    labels = watershed(-distance, markers, mask=cleaned)
    boundaries = find_boundaries(labels, mode='thick')
    final_separated_mask = cleaned & (~boundaries)

    return final_separated_mask, boundaries

def extract_cellular_data(binary_mask, phenotype_name, global_offset_x, global_offset_y):
    """
    Converts the isolated cellular instances into a statistical DataFrame.
    Calculates spatial coordinates and total area for downstream quantitative analysis.
    """
    labeled_mask = label(binary_mask)
    if labeled_mask.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(labeled_mask, properties=('centroid', 'area'))
    df = pd.DataFrame(props)

    df['centroid-0'] += global_offset_y
    df['centroid-1'] += global_offset_x
    df['phenotype'] = phenotype_name

    df.rename(columns={'centroid-0': 'Y_Coord', 'centroid-1': 'X_Coord', 'area': 'Area_Pixels'}, inplace=True)
    return df

def solve_ck_mum1_identity(img_rgb, mask_red_candidate, mask_yellow_candidate):
    """
    Resolves phenotypic overlaps between CK and MUM1 by evaluating the raw RGB signal.
    """
    H, W, _ = img_rgb.shape
    final_ck = np.zeros((H, W), dtype=bool)
    final_mum1 = np.zeros((H, W), dtype=bool)

    candidates = mask_red_candidate | mask_yellow_candidate
    if not np.any(candidates):
        return final_ck, final_mum1

    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)

    green_ratio = G / np.maximum(R, 1.0)

    is_valid = candidates & ~((R < 30) & (G < 30) & (B < 30)) & ~((R > 200) & (G > 200) & (B > 200))
    is_ck = is_valid & (R > 90) & (green_ratio < 0.65) & (R > B * 1.05)
    is_mum1 = is_valid & (~is_ck) & (R > 75) & (G > 70) & (G > (R * 0.55)) & (B < (R * 0.90)) & (R < 195) & (G < 185)

    final_ck[is_ck] = True
    final_mum1[is_mum1] = True

    return final_ck, final_mum1
