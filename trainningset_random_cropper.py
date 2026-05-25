import os
import cv2
import numpy as np
import random
from pathlib import Path

def generate_random_crops(
    input_dir, 
    output_dir, 
    crop_size=(256, 256), 
    num_crops_per_image=10, 
    background_value=-32.0, 
    min_valid_ratio=0.05, 
    max_retries=100,
    delete_original=True
):
    """
    Random crop generator for scientific datasets with random padding logic.
    
    Parameters:
    - input_dir: Path to the original dataset directory (subfolders are preserved).
    - output_dir: Path to save the cropped images.
    - crop_size: Tuple of (Height, Width) for the output crops.
    - num_crops_per_image: Target number of crops to generate per original image.
    - background_value: Value representing the background (e.g., -32.0 for log TIF).
    - min_valid_ratio: Minimum ratio (0.0~1.0) of non-background pixels required in a crop.
    - max_retries: Maximum attempts to find a valid crop per image to avoid infinite loops.
    - delete_original: If True, deletes the original image file after processing.
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
    print(f"Settings: Crop Size {crop_size}, {num_crops_per_image} crops/img, Min Valid Ratio {min_valid_ratio}")
    
    crop_h, crop_w = crop_size
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
        
        # Random padding logic if the original image is smaller than the target crop size
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
            
            if isinstance(background_value, float):
                valid_pixels = np.sum(~np.isclose(crop_img, background_value, atol=1e-5))
            else:
                valid_pixels = np.sum(crop_img != background_value)
                
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

        # Delete the original image if the flag is set
        if delete_original:
            try:
                file_path.unlink()
                print(f"  -> Deleted original file: {file_path.name}")
            except Exception as e:
                print(f"  -> [ERROR] Failed to delete original file {file_path.name}: {e}")

    print("-" * 40)
    print(f"Task completed! Total valid crops generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    
if __name__ == "__main__":
    # ====================================================
    # User Settings
    # ====================================================
    
    INPUT_DIRECTORY = "./normalized_data/train" 
    OUTPUT_DIRECTORY = "./data/train"
    
    TARGET_CROP_SIZE = (512, 512)
    NUM_CROPS = 5
    BG_VAL = 0 
    MIN_FEATURE_RATIO = 0.20 
    
    # Set to True to delete the original image after cropping
    DELETE_ORIGINAL_FILES = False
    
    generate_random_crops(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        crop_size=TARGET_CROP_SIZE,
        num_crops_per_image=NUM_CROPS,
        background_value=BG_VAL,
        min_valid_ratio=MIN_FEATURE_RATIO,
        delete_original=DELETE_ORIGINAL_FILES
    )