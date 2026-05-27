import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from pathlib import Path

def get_gaussian_kernel_2d(kernel_size, sigma):
    """
    Generates a 2D Gaussian convolution kernel.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    kernel_2d = torch.outer(g, g)
    kernel_4d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_4d

def apply_convolutional_resize(image_tensor, scale_factor, kernel_size=3, sigma=1.0, device='cpu'):
    """
    Resizes a 2D image tensor using mathematically strict convolution operations.
    """
    weight = get_gaussian_kernel_2d(kernel_size, sigma).to(device)
    
    if scale_factor < 1.0:
        stride = int(1.0 / scale_factor)
        padding = kernel_size // 2
        
        resized_tensor = F.conv2d(
            image_tensor, 
            weight, 
            stride=stride, 
            padding=padding
        )
        
    elif scale_factor > 1.0:
        stride = int(scale_factor)
        padding = kernel_size // 2
        output_padding = stride - 1 if stride > 1 else 0
        
        resized_tensor = F.conv_transpose2d(
            image_tensor, 
            weight, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        
    else:
        padding = kernel_size // 2
        resized_tensor = F.conv2d(
            image_tensor, 
            weight, 
            stride=1, 
            padding=padding
        )
        
    return resized_tensor

def process_dataset(
    input_dir, 
    output_dir, 
    scale_factor=0.5, 
    kernel_size=3, 
    sigma=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main function to traverse directories, read 0-255 images, apply conv-resize,
    convert back to uint8, and save.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    valid_extensions = {'.tif', '.tiff'}
    all_files = []
    for ext in valid_extensions:
        all_files.extend(input_path.rglob(f"*{ext}"))
        
    if not all_files:
        print(f"No valid TIF files found in {input_dir}.")
        return
        
    print(f"Found {len(all_files)} images.")
    print(f"Device: {device.upper()}")
    print(f"Settings: Scale Factor = {scale_factor}, Kernel Size = {kernel_size}, Sigma = {sigma}")
    
    total_processed = 0
    
    with torch.no_grad():
        for file_path in all_files:
            rel_path = file_path.relative_to(input_path)
            class_folder = rel_path.parent
            
            save_dir = output_path / class_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Read image as uint8 (0-255 values)
            img_np = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img_np is None:
                print(f"[WARNING] Failed to read: {file_path.name}")
                continue
            
            # Convert to float32 for high-precision convolution math
            img_np_float = img_np.astype(np.float32)
            
            # Format to PyTorch Tensor: [Batch, Channel, Height, Width]
            img_tensor = torch.from_numpy(img_np_float).unsqueeze(0).unsqueeze(0).to(device)
            
            # Apply Convolutional Resize in float space
            resized_tensor = apply_convolutional_resize(
                img_tensor, 
                scale_factor=scale_factor, 
                kernel_size=kernel_size, 
                sigma=sigma, 
                device=device
            )
            
            # Retrieve array and remove batch/channel dimensions
            resized_np = resized_tensor.squeeze().cpu().numpy()
            
            # Clamp values to prevent boundary overflow from floating-point rounding
            resized_np = np.clip(resized_np, 0.0, 255.0)
            
            # Convert back to uint8 space with proper rounding
            resized_np_uint8 = np.round(resized_np).astype(np.uint8)
            
            out_filename = f"{file_path.stem}_conv_resize{file_path.suffix}"
            out_filepath = save_dir / out_filename
            
            # Save the final uint8 image
            tifffile.imwrite(str(out_filepath), resized_np_uint8)
            total_processed += 1
            
            if total_processed % 100 == 0:
                print(f"Processed {total_processed}/{len(all_files)} images...")

    print("-" * 40)
    print(f"Task completed! Total resized images: {total_processed}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # ====================================================
    # User Settings
    # ====================================================
    
    INPUT_DIRECTORY = "./normalized_data/train" 
    OUTPUT_DIRECTORY = "./resized_conv_data/train"
    
    # 0.5 = Downsample to half resolution
    # 2.0 = Upsample to double resolution
    SCALE_FACTOR = 0.5 
    
    KERNEL_SIZE = 5 
    GAUSSIAN_SIGMA = 1.0 
    
    COMPUTE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    process_dataset(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        scale_factor=SCALE_FACTOR,
        kernel_size=KERNEL_SIZE,
        sigma=GAUSSIAN_SIGMA,
        device=COMPUTE_DEVICE
    )