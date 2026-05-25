import os
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, test_dir, test_ratio=0.2, move_files=True, random_seed=42):
    """
    Split a dataset into training and testing sets by randomly extracting a percentage of files.
    
    Parameters:
    - source_dir: The directory containing the original training data, organized by class folders.
    - test_dir: The directory where the extracted test data will be saved.
    - test_ratio: The proportion of data to extract for testing (0.0 to 1.0).
    - move_files: If True, moves the files (cut). If False, copies them.
    - random_seed: Seed for the random number generator to ensure reproducibility.
    """
    # Set the random seed so the split is reproducible if needed
    random.seed(random_seed)
    
    src_path = Path(source_dir)
    test_path = Path(test_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
    valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    total_extracted = 0
    
    print(f"Starting dataset split process...")
    print(f"Source: {source_dir}")
    print(f"Target: {test_dir}")
    print(f"Test Ratio: {test_ratio * 100}%")
    print(f"Action: {'Move' if move_files else 'Copy'}")
    print("-" * 50)
    
    # Iterate through each class folder in the source directory
    for class_dir in src_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        target_class_dir = test_path / class_name
        
        # Find all valid image files in the current class folder
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if not files:
            print(f"[WARNING] No valid images found in {class_dir}. Skipping.")
            continue
            
        # Randomly shuffle the files
        random.shuffle(files)
        
        # Calculate the number of files to extract for this class
        num_test = int(len(files) * test_ratio)
        
        # Fallback: if the ratio is too small for a tiny dataset, force at least 1 test image
        if num_test == 0 and len(files) > 1:
            print(f"[WARNING] Class '{class_name}' has only {len(files)} files. Forcing 1 test file.")
            num_test = 1
            
        test_files = files[:num_test]
        
        # Create target class directory if it does not exist
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Move or copy the selected files
        for f in test_files:
            dest_file = target_class_dir / f.name
            if move_files:
                shutil.move(str(f), str(dest_file))
            else:
                shutil.copy2(str(f), str(dest_file))
                
        total_extracted += len(test_files)
        action_str = "Moved" if move_files else "Copied"
        remaining_train = len(files) - len(test_files)
        
        print(f"[INFO] Class '{class_name}': {action_str} {len(test_files)} test files. ({remaining_train} train files remaining)")
        
    print("-" * 50)
    action_str = "moved" if move_files else "copied"
    print(f"Dataset split completed successfully! Total {total_extracted} files {action_str} to {test_dir}.")

if __name__ == "__main__":
    # ====================================================
    # User Settings
    # ====================================================
    
    # Source directory containing the cropped/normalized training data
    SOURCE_DIRECTORY = "./data/train"
    
    # Target directory where the test data will be stored
    TEST_DIRECTORY = "./data/test"
    
    # Ratio of data to extract for testing (0.2 means 20%)
    TEST_RATIO = 0.1
    
    # If True, files are MOVED from train to test (recommended to prevent data leakage).
    # If False, files are COPIED, leaving the original train folder untouched.
    MOVE_FILES = True
    
    # Random seed to ensure you get the exact same split if you run it again with copy mode
    RANDOM_SEED = 42
    
    split_dataset(
        source_dir=SOURCE_DIRECTORY,
        test_dir=TEST_DIRECTORY,
        test_ratio=TEST_RATIO,
        move_files=MOVE_FILES,
        random_seed=RANDOM_SEED
    )