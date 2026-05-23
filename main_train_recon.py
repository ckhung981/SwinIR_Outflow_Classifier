import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import time
from datetime import datetime
import shutil
import random

# Ensure this imports the recognition version of SwinIR
from models.network_swinir_recong import SwinIR 

# ==========================================
# 0. Set Random Seed for Reproducibility
# ==========================================
def set_seed(seed):
    """Fix all random seeds to ensure reproducibility when a specific seed is used."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Force CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# ==========================================
# 1. Dataset
# ==========================================
class ScientificRecognitionDataset(Dataset):
    def __init__(self, root_dir, img_size=128, in_chans=1):
        self.root_dir = root_dir
        self.img_size = img_size
        self.in_chans = in_chans
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        img = img.astype(np.float32) / 255.0 
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.in_chans == 1:
            img = torch.from_numpy(img).unsqueeze(0)
        else:
            img = torch.from_numpy(img).transpose(2, 0, 1)
        return img, label

# ==========================================
# 2. Validation
# ==========================================
def validate(model, test_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    return val_loss / len(test_loader), acc

# ==========================================
# 3. Main
# ==========================================
def main(manual_seed=None):
    # --- Dynamic Seed Generation ---
    
    if manual_seed is None:
        # Generate a random integer seed (0 to 2**32 - 1)
        actual_seed = random.randint(0, 2**32 - 1)
    else:
        actual_seed = manual_seed

    # Apply the seed immediately
    set_seed(actual_seed)

    # --- Hyperparameters ---
    params = {
        "seed": actual_seed, # The generated or manual seed is logged here
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "batch_size": 2,
        "epochs": 150,
        "lr": 2e-4,
        "num_classes": 4,
        "img_size": 512,
        "in_chans": 1,
        "test_every": 10,
        "embed_dim": 96,
        "depths": [6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6]
    }
    
    # --- Timer and Timestamp ---
    start_time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_wall_time = time.time()
    
    # --- Create Run Directory ---
    run_dir = os.path.join('model_weights', start_time_stamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # --- Copy Source Code for Reproducibility ---
    try:
        current_script = __file__
        shutil.copy2(current_script, os.path.join(run_dir, os.path.basename(current_script)))
        
        model_file = 'models/network_swinir_recong.py'
        if os.path.exists(model_file):
            shutil.copy2(model_file, os.path.join(run_dir, 'network_swinir_recong_backup.py'))
        print(f"Source code backed up to: {run_dir}")
    except NameError:
        print("Warning: Could not identify source file for backup.")

    device = torch.device(params["device"])
    
    # --- Save Initial Params to File ---
    info_file = os.path.join(run_dir, 'training_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("--- Hyperparameters ---\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("------------------------\n\n")

    # --- 1. Prepare Data ---
    train_set = ScientificRecognitionDataset('data/train', 
                                             img_size=params["img_size"], in_chans=params["in_chans"])
    test_set = ScientificRecognitionDataset('data/test', 
                                            img_size=params["img_size"], in_chans=params["in_chans"])
    
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False, num_workers=2)
    
    # --- 2. Initialize Model ---
    model = SwinIR(
        img_size=params["img_size"], 
        in_chans=params["in_chans"], 
        num_classes=params["num_classes"],
        window_size=8, 
        depths=params["depths"], 
        embed_dim=params["embed_dim"], 
        num_heads=params["num_heads"],
        mlp_ratio=2, 
        upsampler=''
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Training started. Results will be saved to: {run_dir}")
    
    # --- 3. Training Loop ---
    for epoch in range(params["epochs"]):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if (epoch + 1) % params["test_every"] == 0:
            val_loss, val_acc = validate(model, test_loader, device, criterion)
            print(f">>> [Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
            
            save_path = os.path.join(run_dir, f'model_epoch_{epoch+1}_acc_{val_acc:.1f}.pth')
            torch.save(model.state_dict(), save_path)
            
            with open(info_file, 'a') as f:
                f.write(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%\n")

    # --- Finalize Training Time ---
    end_wall_time = time.time()
    total_duration = end_wall_time - start_wall_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    
    with open(info_file, 'a') as f:
        f.write(f"\nTraining finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {duration_str}\n")
    
    print(f"Training complete. Total time: {duration_str}")

if __name__ == '__main__':
    main(manual_seed=None)