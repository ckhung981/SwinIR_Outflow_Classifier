'''
# --- Inference Script for SwinIR Recognition ---
# This script loads a trained SwinIR model and runs inference on a specified image.
# It dynamically reads model parameters from a training_info.txt file located
# in the same directory as the model weights and outputs the top-3 predictions.
# --- Usage ---
# 1. Set the MODEL_WEIGHTS variable to the path of your trained model weights (.pth file).
# 2. Set the TARGET_IMAGE variable to the path of the image you want to classify.
# 3. Set the CLASSES variable to a list of class names corresponding to your model's output classes.
# 4. Run the script to see the top-3 inference results in the console.
'''

import torch
import cv2
import numpy as np
import os
import ast
from models.network_swinir_recong import SwinIR

def parse_training_info(txt_path):
    """
    Parses the training_info.txt file to extract model parameters.
    Assumes the file contains key-value pairs separated by ':' or '='.
    """
    params = {}
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} not found. Using default parameters.")
        return params

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            # Split by either ':' or '='
            if ':' in line:
                key, val = line.split(':', 1)
            elif '=' in line:
                key, val = line.split('=', 1)
            else:
                continue
            
            key = key.strip()
            val = val.strip()
            
            try:
                # Safely evaluate strings to Python objects (e.g., lists, ints, floats)
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Keep as string if parsing fails (e.g., for upsampler='')
                pass
            params[key] = val
            
    return params

def run_inference(model_path, image_path, classes_list, fallback_in_chans=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Locate training_info.txt in the same directory as the model weights
    model_dir = os.path.dirname(model_path)
    info_path = os.path.join(model_dir, 'training_info.txt')
    
    # Parse parameters from the text file
    parsed_params = parse_training_info(info_path)
    
    # Set default parameters and override with parsed ones if they exist
    model_kwargs = {
        'img_size': parsed_params.get('img_size', 64),
        'in_chans': parsed_params.get('in_chans', fallback_in_chans),
        'num_classes': parsed_params.get('num_classes', len(classes_list)),
        'window_size': parsed_params.get('window_size', 8),
        'depths': parsed_params.get('depths', [6, 6, 6, 6]),
        'embed_dim': parsed_params.get('embed_dim', 96),
        'num_heads': parsed_params.get('num_heads', [6, 6, 6, 6]),
        'mlp_ratio': parsed_params.get('mlp_ratio', 2),
        'upsampler': parsed_params.get('upsampler', '')
    }

    print("Initializing model with the following parameters:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    # 1. Initialize architecture dynamically
    model = SwinIR(**model_kwargs).to(device)

    # 2. Load weights
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Image Preprocessing
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img_input = img.astype(np.float32) / 255.0
    
    # Dynamically determine number of channels based on parsed info
    in_chans = model_kwargs['in_chans']
    
    if in_chans == 1:
        # Handle grayscale format if a colored image was loaded
        if len(img_input.shape) == 3:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    else:
        # Handle RGB format if a grayscale image was loaded
        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    
    img_tensor = img_tensor.to(device)

    # 4. Run Inference
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = model(img_tensor)
        else:
            output = model(img_tensor)
            
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top-k predictions (up to 3, but handles cases where num_classes < 3)
        k = min(3, len(classes_list))
        topk_confs, topk_indices = torch.topk(probabilities, k=k, dim=1)
        
        # Convert tensors to lists for easier handling
        topk_confs_list = topk_confs[0].tolist()
        topk_indices_list = topk_indices[0].tolist()

    return topk_indices_list, topk_confs_list

if __name__ == '__main__':
    MODEL_WEIGHTS = 'model_weights/20260526_132921/model_epoch_45_acc_31.2.pth' 
    TARGET_IMAGE = 'normalized_data/train/08_n2_m90/output_n2_m90_1bp_150deg_raw.tif'
    CLASSES = ['n1_m6','n1_m30','n1_m60','n1_m90','n2_m6','n2_m30','n2_m60','n2_m90'
            ,'n4_m6','n4_m30','n4_m60','n4_m90','n6_m6','n6_m30','n6_m60','n6_m90']
    
    try:
        topk_indices, topk_confs = run_inference(
            model_path=MODEL_WEIGHTS,
            image_path=TARGET_IMAGE,
            classes_list=CLASSES,
            fallback_in_chans=1
        )
        
        print("-" * 40)
        print("Inference Results:")
        print(f"File: {os.path.basename(TARGET_IMAGE)}")
        print("Top Predictions:")
        
        for rank, (idx, conf) in enumerate(zip(topk_indices, topk_confs), start=1):
            print(f"  {rank}. {CLASSES[idx]} (Index: {idx}) - Confidence: {conf*100:.2f}%")
            
        print("-" * 40)
        
    except Exception as e:
        print(f"Inference failed: {e}")