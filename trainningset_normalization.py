'''
This script is designed to process a dataset of .tif/.tiff images for
training a recognition model.
The main functionalities include:
1. Classifying images into categories based on their filenames using a user-defined mapping.
2. Rescaling pixel values of the images to a range of 1-255 (with 0 reserved for background) based on specified lower and upper percentile thresholds.
3. Generating histograms of pixel values for each image, with visual indicators for the percentile thresholds.
4. Creating preview images that visually differentiate pixels below the lower threshold (red), above the upper threshold (blue), and those within the thresholds (grayscale).
5. Organizing the processed images and previews into structured directories for training and diagnostic purposes.
'''





import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ==========================================
# 使用者自定義：檔名對應分類類別的規則
# ==========================================
# key: 檔名(或檔名中包含的關鍵字)
# value: 對應的分類資料夾名稱
CATEGORY_MAPPING = {
    "n1_m6_": "01_n1_m6",
    "n1_m30_": "02_n1_m30",
    "n1_m60_": "03_n1_m60",
    "n1_m90_": "04_n1_m90",
    "n2_m6_": "05_n2_m6",
    "n2_m30_": "06_n2_m30",
    "n2_m60_": "07_n2_m60",
    "n2_m90_": "08_n2_m90",
    "n4_m6_": "09_n4_m6",
    "n4_m30_": "10_n4_m30",
    "n4_m60_": "11_n4_m60",
    "n4_m90_": "12_n4_m90",
    "n6_m6_": "13_n6_m6",
    "n6_m30_": "14_n6_m30",
    "n6_m60_": "15_n6_m60",
    "n6_m90_": "16_n6_m90",              
    # 可以在此繼續新增您的自定義規則，例如：
    # "outflow_test_01": "05_test",
}

def get_category_from_filename(filename, mapping):
    """
    根據檔名回傳對應的分類類別。
    比對邏輯：只要檔名(不含副檔名)包含了 mapping 中的 key，就回傳對應的 value。
    """
    file_base = os.path.splitext(filename)[0]
    
    # 尋找檔名是否包含我們定義的關鍵字
    for key, category in mapping.items():
        if key in file_base:
            return category
            
    # 如果都沒有匹配到，預設放入一個未分類資料夾
    return "uncategorized"

def process_and_categorize_datasets(source_dir, train_dir, preview_dir, low_p, high_p, mapping):
    """
    讀取 source_dir 底下的所有 tif/tiff，依照使用者定義的檔名對應邏輯進行分類。
    執行 rescale，繪製 histogram，製作 preview，並分別存入 train_dir 與 preview_dir。
    """
    # 遞迴尋找所有 tif 與 tiff 檔案
    files = glob.glob(os.path.join(source_dir, "**/*.tif"), recursive=True) + \
            glob.glob(os.path.join(source_dir, "**/*.tiff"), recursive=True)
    
    if not files:
        print(f"在 {source_dir} 中找不到任何 .tif / .tiff 檔案。")
        return

    print(f"\n[處理中] 使用百分位數截斷: 下限 {low_p}%, 上限 {high_p}%")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        file_base = os.path.splitext(filename)[0]
        
        # 依照檔名取得使用者自定義的類別
        category = get_category_from_filename(filename, mapping)
        
        # 建立目標資料夾路徑
        cat_train_dir = os.path.join(train_dir, category)
        cat_preview_dir = os.path.join(preview_dir, category)
        
        # 自動建立不存在的資料夾
        os.makedirs(cat_train_dir, exist_ok=True)
        os.makedirs(cat_preview_dir, exist_ok=True)
        
        # 讀取影像
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: 
            print(f"無法讀取: {file_path}")
            continue
            
        h, w = img.shape[:2]
        img_float = img.astype(np.float32)
        
        # 定義有效像素的 Mask (背景值為 -32 不算在內)
        valid_mask = img_float != -32
        
        if not np.any(valid_mask):
            print(f"全為無效值 (-32): {file_path}")
            continue
            
        valid_pixels = img_float[valid_mask]
        
        # 計算截斷範圍
        vmin = np.percentile(valid_pixels, low_p)
        vmax = np.percentile(valid_pixels, 100 - high_p)
        
        # 1. 製作並儲存 Histogram
        hist_out_path = os.path.join(cat_preview_dir, f"{file_base}_hist.png")
        plt.figure(figsize=(8, 4))
        plt.hist(valid_pixels, bins=100, color='blue', alpha=0.7)
        if vmax > vmin:
            plt.axvline(vmin, color='red', linestyle='dashed', linewidth=1.5, label=f'Low {low_p}%: {vmin:.2e}')
            plt.axvline(vmax, color='green', linestyle='dashed', linewidth=1.5, label=f'High {100-high_p}%: {vmax:.2e}')
        plt.title(f"Pixel Value Histogram - {filename}")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(hist_out_path)
        plt.close()
        
        # 2. 製作並儲存 Final TIF (Rescale 到 1-255 灰階，0 保留給背景)
        # 初始化全為 0 的陣列，所以非 valid_mask 的地方(背景)會維持為 0
        final_img = np.zeros((h, w), dtype=np.uint8)
        
        if vmax > vmin:
            # 限制數值範圍在 vmin 到 vmax 之間
            clipped_data = np.clip(img_float, vmin, vmax)
            
            # 【核心修改處】縮放至 1 ~ 255 的範圍
            # 計算公式: (目前值 - 最小值) / (最大值 - 最小值) * 254.0 + 1.0
            normalized_data = (clipped_data - vmin) / (vmax - vmin) * 254.0 + 1.0
            
            # 四捨五入後，僅將有效像素寫入 final_img
            final_img[valid_mask] = np.round(normalized_data)[valid_mask].astype(np.uint8)
            
        elif vmax == vmin:
            # 若所有像素值相同，則將所有有效像素設為 1 (最低有效值)
            final_img[valid_mask] = 1
            
        final_tif_path = os.path.join(cat_train_dir, filename)
        cv2.imwrite(final_tif_path, final_img)
        
        # 3. 製作並儲存 Preview PNG
        preview_img = np.zeros((h, w, 3), dtype=np.uint8)
        below_mask = (img_float < vmin) & valid_mask
        above_mask = (img_float > vmax) & valid_mask
        inside_mask = (img_float >= vmin) & (img_float <= vmax) & valid_mask
        
        if vmax >= vmin:
            # 灰階部分會直接取 final_img 的值 (現在是 1~255)
            gray_part = final_img[inside_mask]
            preview_img[inside_mask] = np.stack([gray_part]*3, axis=-1)
            
        preview_img[below_mask] = [0, 0, 255] # BGR: 顯示為紅色 (低於下限)
        preview_img[above_mask] = [255, 0, 0] # BGR: 顯示為藍色 (高於上限)
        
        preview_out_path = os.path.join(cat_preview_dir, f"{file_base}_preview.png")
        cv2.imwrite(preview_out_path, preview_img)
        
        print(f"已完成: 分類 [{category}] -> {filename} | 範圍: {vmin:.2e} ~ {vmax:.2e}")

if __name__ == "__main__":
    # --- 參數設定區 ---
    SOURCE_DIR = "./raw_data/data_from_synline"  # 原始資料夾路徑，請自行修改
    TRAIN_DIR = "normalized_data/train"
    PREVIEW_DIR = "data/trainningset_preview"
    
    LOWER_PERCENT = 2.0
    UPPER_PERCENT = 0.0
    # ------------------

    print("Step 1: 開始執行自定義分類、Rescale(1-255)、Histogram 與 Preview 生成...")
    process_and_categorize_datasets(SOURCE_DIR, TRAIN_DIR, PREVIEW_DIR, LOWER_PERCENT, UPPER_PERCENT, CATEGORY_MAPPING)
    
    print("\n[任務完成]")
    print(f"1. 訓練用的 TIF 檔案已依照檔名規則分類存放於: {TRAIN_DIR}")
    print(f"2. 診斷用的 Preview 與 Histogram 存放於: {PREVIEW_DIR}")