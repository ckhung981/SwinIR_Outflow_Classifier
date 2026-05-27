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
import gc
import re

# Ensure this imports the recognition version of SwinIR
from models.network_swinir_recong import SwinIR 

def clear_gpu_memory():
    """
    Forces garbage collection and clears CUDA cache, 
    then prints the current VRAM usage to verify it is clean.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print("=" * 40)
        print("GPU Memory Status After Clearing:")
        print(f"Allocated: {allocated_memory:.2f} MB")
        print(f"Reserved:  {reserved_memory:.2f} MB")
        print("=" * 40)
        
        if allocated_memory > 100:
            print("[WARNING] GPU memory is not completely clean. Check for zombie processes.")

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
    def __init__(self, root_dir, in_chans=1):
        self.root_dir = root_dir
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
            img = np.zeros((8, 8), dtype=np.float32)
            
        img = img.astype(np.float32) / 255.0 
        
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
    clear_gpu_memory()
    
    # --- Dynamic Seed Generation ---
    if manual_seed is None:
        actual_seed = random.randint(0, 2**32 - 1)
    else:
        actual_seed = manual_seed

    # Apply the seed immediately
    set_seed(actual_seed)

    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise EnvironmentError("CUDA is not available. Please check your GPU setup.")
        
    # --- Training Parameters ---
    params = {
        "seed": actual_seed, 
        "device": 'cuda',
        "batch_size": 3,
        "start_epoch": 0,      # Default value, will be auto-updated if resuming
        "epochs": 100,         
        "lr": 1e-4,
        "num_classes": 16,
        "in_chans": 1,
        "test_every": 5,
        "embed_dim": 32,
        "depths": [3, 3, 3],
        "num_heads": [4, 4, 4],
        
        # --- Resume Training Settings ---
        "resume_path": "model_weights/20260527_083416/model_epoch_60_acc_31.2.pth", # Path to the checkpoint to resume training
        "resume_append_dir": False # Whether to append to the existing run directory when resuming
    }
    
    start_wall_time = time.time()
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    device = torch.device(params["device"])
    
    # --- Determine Run Directory and Auto-detect Epoch ---
    is_resuming = params["resume_path"] is not None and os.path.exists(params["resume_path"])
    
    if is_resuming and params["resume_append_dir"]:
        # Extract the directory of the existing checkpoint
        run_dir = os.path.dirname(params["resume_path"])
        print(f"Appending to existing run directory: {run_dir}")
        
        # Auto-detect epoch from filename using regex
        filename = os.path.basename(params["resume_path"])
        match = re.search(r'model_epoch_(\d+)', filename)
        if match:
            params["start_epoch"] = int(match.group(1))
            print(f"Auto-detected starting epoch: {params['start_epoch']} (Next training epoch will be {params['start_epoch'] + 1})")
        else:
            print("[WARNING] Could not auto-detect epoch from filename. Using default start_epoch.")
    else:
        # Create a completely new directory
        run_dir = os.path.join('model_weights', current_time_str)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run directory: {run_dir}")
    
    info_file = os.path.join(run_dir, 'training_info.txt')
    
    # --- Source Code Backup (Only if creating a new directory) ---
    if not (is_resuming and params["resume_append_dir"]):
        try:
            current_script = __file__
            shutil.copy2(current_script, os.path.join(run_dir, os.path.basename(current_script)))
            model_file = 'models/network_swinir_recong.py'
            if os.path.exists(model_file):
                shutil.copy2(model_file, os.path.join(run_dir, 'network_swinir_recong_backup.py'))
        except NameError:
            print("Warning: Could not identify source file for backup.")
            
        # Write initial parameter log
        with open(info_file, 'w') as f:
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("--- Hyperparameters ---\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            f.write("------------------------\n\n")

    # Log resume action
    if is_resuming:
        with open(info_file, 'a') as f:
            f.write(f"\n--- Resumed Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Loaded Checkpoint: {params['resume_path']}\n")
            f.write(f"Starting from Epoch: {params['start_epoch']}\n\n")

    # --- 1. Prepare Data ---
    train_set = ScientificRecognitionDataset('data/train', in_chans=params["in_chans"])
    test_set = ScientificRecognitionDataset('data/test', in_chans=params["in_chans"])
    
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2) 
    
    # --- 2. Initialize Model ---
    model = SwinIR(
        img_size=64, 
        in_chans=params["in_chans"], 
        num_classes=params["num_classes"],
        window_size=8, 
        depths=params["depths"], 
        embed_dim=params["embed_dim"], 
        num_heads=params["num_heads"],
        mlp_ratio=2, 
        upsampler=''
    ).to(device)

    # --- 2.5 Load Resume Checkpoint ---
    if is_resuming:
        print(f"Loading weights from checkpoint: {params['resume_path']}")
        model.load_state_dict(torch.load(params["resume_path"], map_location=device))
        print("Checkpoint loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize ReduceLROnPlateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6
    )
    
    print(f"Training execution starts. Results will be saved to: {run_dir}")
    
# --- 3. Training Loop ---
    # The loop will start from start_epoch and continue until epochs
    for epoch in range(params["start_epoch"], params["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation and Checkpoint Saving
        if (epoch + 1) % params["test_every"] == 0:
            val_loss, val_acc = validate(model, test_loader, device, criterion)
            
            # Fetch current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f">>> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch+1}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")
            
            save_path = os.path.join(run_dir, f'model_epoch_{epoch+1}_acc_{val_acc:.1f}.pth')
            torch.save(model.state_dict(), save_path)
            
            with open(info_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}: "
                        f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr:.2e}\n")
                
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f">>> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch+1}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e}")

    # --- Finalize Training Time ---
    end_wall_time = time.time()
    total_duration = end_wall_time - start_wall_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    
    with open(info_file, 'a') as f:
        f.write(f"\nTraining execution finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Duration: {duration_str}\n")
    
    print(f"Training complete. Session time: {duration_str}")

if __name__ == '__main__':
    main(manual_seed=None)
