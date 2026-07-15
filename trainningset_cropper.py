import os
import cv2
import numpy as np
import random
import math
import re
from pathlib import Path

def generate_crops(
    input_dir, 
    output_dir, 
    crop_mode='random', 
    crop_size=(256, 256), 
    num_crops_per_image=10, 
    fixed_crop_config=None, 
    background_value=0, 
    min_valid_ratio=0.05, 
    max_retries=100,
    delete_original=False
):
    """
    Image crop generator for scientific datasets supporting both random and dynamic fixed regions.
    The fixed region dynamically scales the Y-axis steps based on the inclination angle 
    extracted from the filename, applying a sin(theta) projection correction.
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    valid_extensions = {'.tif', '.tiff', '.png'}
    
    all_files = []
    for ext in valid_extensions:
        all_files.extend(input_path.rglob(f"*{ext}"))
        
    if not all_files:
        print(f"No valid image files found in {input_dir}.")
        return

    print(f"Found {len(all_files)} original images. Starting cropping process...")
    print(f"Mode: {crop_mode.upper()}")
    
    if crop_mode == 'random':
        print(f"Settings: Crop Size {crop_size}, {num_crops_per_image} crops/img, Min Valid Ratio {min_valid_ratio}")
    else:
        print(f"Settings: Dynamic Fixed Mode enabled. Origin: Bottom-Center.")
        print(f"Base Configuration: {fixed_crop_config}")
    
    total_generated = 0
    
    for file_path in all_files:
        rel_path = file_path.relative_to(input_path)
        class_folder = rel_path.parent
        
        save_dir = output_path / class_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARNING] Failed to read image, skipping: {file_path}")
            continue
            
        h, w = img.shape[:2]
        
        if crop_mode == 'fixed':
            # 1. Get base config
            ux = fixed_crop_config['x']
            fw = fixed_crop_config['w']
            fh = fixed_crop_config['h']
            base_y_steps = fixed_crop_config['base_y_steps']
            
            # 2. Extract angle from filename (default 90)
            angle = 90.0 
            match = re.search(r'_(\d+)deg_', file_path.name)
            if match:
                angle = float(match.group(1))
                
            # 3. Calculate 2D projection scale
            projection_scale = abs(math.sin(math.radians(angle)))
            
            # Iterate all bounding boxes
            for box_idx, base_y in enumerate(base_y_steps):
                # 4. Dynamically adjust Y start point
                uy = int(base_y * projection_scale)

                fx = int((w // 2) + ux)      
                fy = int(h - uy - fh)         

                if len(img.shape) == 3:
                    crop_img = np.full((fh, fw, img.shape[2]), background_value, dtype=img.dtype)
                else:
                    crop_img = np.full((fh, fw), background_value, dtype=img.dtype)

                x1_img, y1_img = max(0, fx), max(0, fy)
                x2_img, y2_img = min(w, fx + fw), min(h, fy + fh)
                
                x1_crop, y1_crop = max(0, -fx), max(0, -fy)
                x2_crop = x1_crop + (x2_img - x1_img)
                y2_crop = y1_crop + (y2_img - y1_img)
                
                if x1_img < x2_img and y1_img < y2_img:
                    crop_img[y1_crop:y2_crop, x1_crop:x2_crop] = img[y1_img:y2_img, x1_img:x2_img]
                    
                out_filename = f"{file_path.stem}_fixed_{box_idx:02d}{file_path.suffix}"
                out_filepath = save_dir / out_filename
                cv2.imwrite(str(out_filepath), crop_img)
                total_generated += 1
                
            print(f"[SUCCESS] {file_path.name} -> Angle: {angle}°, Scale: {projection_scale:.2f} -> Generated {len(base_y_steps)} crops.")

        elif crop_mode == 'random':
            crop_h, crop_w = crop_size
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
            
            if h < crop_h:
                total_pad_h = crop_h - h
                pad_top = random.randint(0, total_pad_h)
                pad_bottom = total_pad_h - pad_top
                
            if w < crop_w:
                total_pad_w = crop_w - w
                pad_left = random.randint(0, total_pad_w)
                pad_right = total_pad_w - pad_left
                
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                img = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=background_value
                )
                h, w = img.shape[:2]

            crops_found = 0
            retries = 0
            
            while crops_found < num_crops_per_image and retries < max_retries:
                x_start = random.randint(0, w - crop_w)
                y_start = random.randint(0, h - crop_h)
                
                crop_img = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
                
                crop_img_float = crop_img.astype(np.float32)
                
                if isinstance(background_value, float):
                    valid_pixels = np.sum(~np.isclose(crop_img_float, background_value, atol=1e-5))
                else:
                    valid_pixels = np.sum(crop_img_float != float(background_value))
                    
                valid_ratio = valid_pixels / (crop_h * crop_w)
                
                if valid_ratio >= min_valid_ratio:
                    out_filename = f"{file_path.stem}_crop_{crops_found+1:03d}{file_path.suffix}"
                    out_filepath = save_dir / out_filename
                    
                    cv2.imwrite(str(out_filepath), crop_img)
                    crops_found += 1
                    total_generated += 1
                else:
                    retries += 1
                    
            if crops_found < num_crops_per_image:
                print(f"[INFO] {file_path.name}: Found {crops_found}/{num_crops_per_image} valid crops after {max_retries} retries.")
            else:
                print(f"[SUCCESS] {file_path.name} -> Generated {crops_found} crops.")

        if delete_original:
            try:
                file_path.unlink()
                print(f"  -> Deleted original file: {file_path.name}")
            except Exception as e:
                print(f"  -> [ERROR] Failed to delete original file {file_path.name}: {e}")

    print("-" * 40)
    print(f"Task completed! Total valid crops generated: {total_generated}")
    print(f"Output directory: {output_dir}")

def pad_to_max_size(input_dir, output_dir, background_value=0):
    """
    Scans the entire directory to find the global maximum width and height.
    Then applies centered padding to all images to match this maximum shape.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    valid_extensions = {'.tif', '.tiff', '.png'}
    all_files = []
    for ext in valid_extensions:
        all_files.extend(input_path.rglob(f"*{ext}"))
        
    if not all_files:
        print(f"No valid image files found in {input_dir}.")
        return

    # Pass 1: Find global maximum dimensions
    print(f"Found {len(all_files)} images. Pass 1: Scanning for global maximum dimensions...")
    max_h, max_w = 0, 0
    for file_path in all_files:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        
    print(f"Global max dimensions found: Width = {max_w}, Height = {max_h}")
    print("-" * 40)
    print("Pass 2: Applying centered padding to all images...")
    
    total_processed = 0
    # Pass 2: Apply centered padding
    for file_path in all_files:
        rel_path = file_path.relative_to(input_path)
        class_folder = rel_path.parent
        
        save_dir = output_path / class_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARNING] Failed to read image, skipping: {file_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Calculate required padding
        delta_w = max_w - w
        delta_h = max_h - h
        
        # Distribute padding equally to center the image
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_img = cv2.copyMakeBorder(
                img, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=background_value
            )
        else:
            padded_img = img
            
        out_filepath = save_dir / file_path.name
        cv2.imwrite(str(out_filepath), padded_img)
        total_processed += 1
        
        print(f"[SUCCESS] {file_path.name} -> Padded from ({w}x{h}) to ({max_w}x{max_h})")

    print("-" * 40)
    print(f"Task completed! Total padded images saved: {total_processed}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # ====================================================
    # User Settings
    # ====================================================
    
    # Choose operation mode: 'crop' or 'pad_to_max'
    OPERATION_MODE = 'pad_to_max' 
    
    INPUT_DIRECTORY = "./resized_conv_data/validation_4"  # Directory containing original images
    OUTPUT_DIRECTORY = "./data/validation_4"  # Directory to save processed images
    
    BG_VAL = 0 
    
    if OPERATION_MODE == 'pad_to_max':
        # Execute global maximum size centering & padding
        pad_to_max_size(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            background_value=BG_VAL
        )
        
    elif OPERATION_MODE == 'crop':
        # Execute original cropping logic
        CROP_MODE = 'fixed' # 'random' or 'fixed'
        
        # For dynamic fixed mode
        DYNAMIC_FIXED_CONFIG = {
            'x': -128,
            'w': 256,
            'h': 256,
            'base_y_steps': [0,32,64]
        }
        
        # For random mode
        TARGET_CROP_SIZE = (256, 256)
        NUM_CROPS = 3
        MIN_FEATURE_RATIO = 0.05
        
        DELETE_ORIGINAL_FILES = False
        
        generate_crops(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            crop_mode=CROP_MODE,
            crop_size=TARGET_CROP_SIZE,
            num_crops_per_image=NUM_CROPS,
            fixed_crop_config=DYNAMIC_FIXED_CONFIG,
            background_value=BG_VAL,
            min_valid_ratio=MIN_FEATURE_RATIO,
            delete_original=DELETE_ORIGINAL_FILES
        )
    else:
        print("Invalid OPERATION_MODE. Please choose 'crop' or 'pad_to_max'.")