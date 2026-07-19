import os
import time
import gc
import json
import torch
import pyvips
import numpy as np
import pandas as pd
import skimage.morphology as morph

# Importiamo le funzioni matematiche che abbiamo isolato
from utils.morphology import refine_mask, extract_cellular_data, solve_ck_mum1_identity

# Configurazione PyVIPS per evitare Out-Of-Memory
pyvips.cache_set_max(0)

# Parametri geometrici e biologici
VIEW_SIZE = 1024
PADDING = 128
MIN_CELL_SIZE = 2000

COLOR_MAP = {
    'MPO':   [139, 69, 19],    'CD11b': [0, 200, 0],
    'CD8':   [0, 255, 255],    'SMA':   [138, 43, 226],
    'MUM1':  [255, 215, 0],    'CK':    [255, 0, 0],
    'FoxP3': [0, 0, 0],        'BG':    [255, 255, 255]
}

def process_slide(tiff_path, model_A, model_B, input_root, output_root, device):
    """
    Motore di inferenza principale. Coordina I/O su disco, tiling e inferenza neurale.
    """
    rel_path = os.path.relpath(tiff_path, input_root)
    out_dir = os.path.dirname(os.path.join(output_root, rel_path))
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    base_name = os.path.basename(tiff_path).replace('.tiff', '')
    out_path = os.path.join(out_dir, base_name + '_ANALYSIS_V6.tiff')
    csv_path = os.path.join(out_dir, base_name + '_CELL_DATA.csv')

    temp_bin = os.path.join(out_dir, f"TEMP_{base_name}.bin")
    meta_json = os.path.join(out_dir, f"TEMP_{base_name}.json")

    if os.path.exists(out_path) and os.path.exists(csv_path):
        print(f"Skipping (already processed): {base_name}")
        return

    print(f"Processing: {base_name} (Tile {VIEW_SIZE}, Padding {PADDING})")
    start_time = time.time()
    slide_statistics = []

    try:
        img = pyvips.Image.new_from_file(tiff_path)
        W, H = img.width, img.height

        with open(meta_json, 'w') as f:
            json.dump({'width': W, 'height': H}, f)

        vis_img = np.memmap(temp_bin, dtype=np.uint8, mode='w+', shape=(H, W, 3))
        vis_img[:] = 255
        vis_img.flush()

        for y in range(0, H, VIEW_SIZE):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for x in range(0, W, VIEW_SIZE):
                try:
                    x0 = max(0, x - PADDING)
                    y0 = max(0, y - PADDING)
                    x1 = min(W, x + VIEW_SIZE + PADDING)
                    y1 = min(H, y + VIEW_SIZE + PADDING)

                    crop_w = x1 - x0
                    crop_h = y1 - y0

                    tile = img.crop(x0, y0, crop_w, crop_h)
                    mem = tile.write_to_memory()
                    tile_arr = np.frombuffer(mem, dtype=np.uint8).reshape(tile.height, tile.width, -1)[:,:,:3]

                    if tile_arr.mean() > 245 or tile_arr.mean() < 10:
                        continue

                    inp = tile_arr.astype(np.float32) / 255.0
                    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred_A = model_A(inp_tensor).squeeze(0).cpu().numpy()
                        pred_B = model_B(inp_tensor).squeeze(0).cpu().numpy()

                    mask_mpo      = (pred_B[0] > 0.4)
                    mask_foxp3    = (pred_B[2] > 0.4)
                    mask_cd_raw   = (pred_B[3] > 0.3)
                    mask_mum1_raw = (pred_B[4] > 0.5)
                    mask_ck_raw   = (pred_A[5] > 0.55)
                    mask_sma_raw  = (pred_A[1] > 0.25)

                    fixed_ck, fixed_mum1 = solve_ck_mum1_identity(tile_arr, mask_ck_raw, mask_mum1_raw)

                    mask_cd8 = np.zeros_like(mask_cd_raw, dtype=bool)
                    mask_cd11b = np.zeros_like(mask_cd_raw, dtype=bool)
                    if np.any(mask_cd_raw):
                        b_chan = tile_arr[:,:,2].astype(np.float32)
                        g_chan = tile_arr[:,:,1].astype(np.float32)
                        r_chan = tile_arr[:,:,0].astype(np.float32)
                        
                        is_cd8 = mask_cd_raw & (b_chan > r_chan) & (b_chan > (g_chan * 0.85))
                        mask_cd8[is_cd8] = True
                        mask_cd11b[mask_cd_raw & (~is_cd8)] = True

                    fixed_mum1 = fixed_mum1 & (~mask_sma_raw)
                    fixed_ck = fixed_ck & (~fixed_mum1)
                    mask_cd11b = mask_cd11b & (~fixed_ck) & (~fixed_mum1)
                    mask_cd8   = mask_cd8 & (~fixed_ck) & (~fixed_mum1) & (~mask_cd11b)
                    mask_mpo   = mask_mpo & (~fixed_ck) & (~fixed_mum1) & (~mask_cd11b) & (~mask_cd8)
                    mask_foxp3 = mask_foxp3 & (~fixed_ck) & (~fixed_mum1) & (~mask_cd11b) & (~mask_cd8) & (~mask_mpo)

                    immune_active = fixed_mum1 | mask_cd11b | mask_cd8 | mask_mpo | mask_foxp3
                    fixed_sma = mask_sma_raw & (~fixed_ck) & (~immune_active)

                    fixed_cd8, bound_cd8    = refine_mask(mask_cd8, min_size=MIN_CELL_SIZE, closing_disk=20, peak_min_dist=70)
                    mask_foxp3, bound_foxp3 = refine_mask(mask_foxp3, min_size=MIN_CELL_SIZE, closing_disk=5, peak_min_dist=12)
                    
                    mask_mpo, bound_mpo     = refine_mask(mask_mpo, min_size=MIN_CELL_SIZE, closing_disk=10, peak_min_dist=15)
                    
                    final_mum1_refined, bound_mum1 = refine_mask(fixed_mum1, min_size=100, closing_disk=15, peak_min_dist=35)
                    mask_cd11b, bound_cd11b = refine_mask(mask_cd11b, min_size=MIN_CELL_SIZE, closing_disk=10, peak_min_dist=20)
                    
                    all_boundaries = bound_mpo | bound_foxp3 | bound_cd11b | bound_mum1 | bound_cd8

                    fixed_ck  = morph.remove_small_objects(fixed_ck, min_size=200)
                    fixed_sma = morph.binary_closing(fixed_sma, morph.disk(4))
                    fixed_sma = morph.remove_small_objects(fixed_sma, min_size=80)

                    valid_x_start = x - x0
                    valid_y_start = y - y0
                    valid_x_end = valid_x_start + min(VIEW_SIZE, W - x)
                    valid_y_end = valid_y_start + min(VIEW_SIZE, H - y)

                    final_sma   = fixed_sma[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_ck    = fixed_ck[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_mpo   = mask_mpo[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_cd11b = mask_cd11b[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_cd8   = fixed_cd8[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_mum1  = final_mum1_refined[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_foxp3 = mask_foxp3[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    final_boundaries = all_boundaries[valid_y_start:valid_y_end, valid_x_start:valid_x_end]

                    write_w = valid_x_end - valid_x_start
                    write_h = valid_y_end - valid_y_start
                    roi_target = vis_img[y:y+write_h, x:x+write_w]

                    if np.any(final_sma):   roi_target[final_sma]   = COLOR_MAP['SMA']
                    if np.any(final_ck):    roi_target[final_ck]    = COLOR_MAP['CK']
                    if np.any(final_mpo):   roi_target[final_mpo]   = COLOR_MAP['MPO']
                    if np.any(final_cd11b): roi_target[final_cd11b] = COLOR_MAP['CD11b']
                    if np.any(final_cd8):   roi_target[final_cd8]   = COLOR_MAP['CD8']
                    if np.any(final_mum1):  roi_target[final_mum1]  = COLOR_MAP['MUM1']
                    if np.any(final_foxp3): roi_target[final_foxp3] = COLOR_MAP['FoxP3']
                    if np.any(final_boundaries): roi_target[final_boundaries] = [0, 0, 0]

                    slide_statistics.append(extract_cellular_data(final_mpo, 'MPO', x, y))
                    slide_statistics.append(extract_cellular_data(final_cd11b, 'CD11b', x, y))
                    slide_statistics.append(extract_cellular_data(final_cd8, 'CD8', x, y))
                    slide_statistics.append(extract_cellular_data(final_mum1, 'MUM1', x, y))
                    slide_statistics.append(extract_cellular_data(final_foxp3, 'FoxP3', x, y))

                    if np.any(final_ck):
                        df_ck = pd.DataFrame({'Y_Coord': [y], 'X_Coord': [x], 'Area_Pixels': [final_ck.sum()], 'phenotype': ['CK_Area']})
                        slide_statistics.append(df_ck)

                    if np.any(final_sma):
                        df_sma = pd.DataFrame({'Y_Coord': [y], 'X_Coord': [x], 'Area_Pixels': [final_sma.sum()], 'phenotype': ['SMA_Area']})
                        slide_statistics.append(df_sma)

                except Exception as e_inner:
                    print(f"Error processing tile {x},{y}: {e_inner}")
                    continue

            vis_img.flush()

        print(f"Finalizing outputs...")
        del vis_img
        gc.collect()

        raw_arr = np.memmap(temp_bin, dtype=np.uint8, mode='r', shape=(H,W,3))
        vips_out = pyvips.Image.new_from_array(raw_arr)
        vips_out.write_to_file(out_path, compression="lzw", bigtiff=True, tile=True, tile_width=512, tile_height=512, pyramid=True)

        if slide_statistics:
            final_df = pd.concat([df for df in slide_statistics if not df.empty], ignore_index=True)
            final_df.to_csv(csv_path, index=False)

        del raw_arr, vips_out
        os.remove(temp_bin)
        os.remove(meta_json)

        elapsed = time.time() - start_time
        print(f"Completed: {out_path} and {csv_path} ({elapsed:.1f}s)")

    except Exception as e:
        print(f"CRITICAL ERROR on {base_name}: {e}")
