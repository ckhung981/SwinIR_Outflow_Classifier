'''
# --- Inference Script for SwinIR Recognition ---
# This script loads a trained SwinIR model and runs inference on a specified image.
# It outputs the predicted class and confidence score for the input image.
# --- Usage ---
# 1. Set the MODEL_WEIGHTS variable to the path of your trained model weights (.
    pth file).
# 2. Set the TARGET_IMAGE variable to the path of the image you want to classify.
# 3. Set the CLASSES variable to a list of class names corresponding to your model's output classes.
# 4. Run the script to see the inference results in the console.
'''



import torch
import cv2
import numpy as np
import os
from models.network_swinir_recong import SwinIR

def run_inference(model_path, image_path, num_classes, in_chans=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize architecture (must match training settings)
    model = SwinIR(
        img_size=64,  # Dummy value, network dynamically pads via check_image_size
        in_chans=in_chans, 
        num_classes=num_classes,
        window_size=8, 
        depths=[6, 6, 6, 6], 
        embed_dim=96, 
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler=''
    ).to(device)

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
    
    # Removed cv2.resize to keep the original physical aspect ratio
    
    if in_chans == 1:
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    else:
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
        conf, predicted = torch.max(probabilities, 1)

    return predicted.item(), conf.item()

if __name__ == '__main__':
    # --- Settings ---
    MODEL_WEIGHTS = '/content/model_weights/20260511_062329/model_epoch_150_acc_100.0.pth' 
    TARGET_IMAGE = '/content/drive/MyDrive/transformer/data/test/01_n1/output_n1_m6_1bp_90deg_range0.5.tif'
    CLASSES = ['n1', 'n2', 'n4', 'n6'] 
    
    try:
        class_idx, confidence = run_inference(
            model_path=MODEL_WEIGHTS,
            image_path=TARGET_IMAGE,
            num_classes=len(CLASSES),
            in_chans=1
        )
        
        print("-" * 40)
        print("Inference Results:")
        print(f"File: {os.path.basename(TARGET_IMAGE)}")
        print(f"Predicted: {CLASSES[class_idx]} (Index: {class_idx})")
        print(f"Confidence: {confidence*100:.2f}%")
        print("-" * 40)
        
    except Exception as e:
        print(f"Inference failed: {e}")