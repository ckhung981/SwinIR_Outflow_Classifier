#!/bin/bash

# ==========================================
# 參數設定區 (請依照您的實際環境修改)
# ==========================================
# 1. 您的 kawas 帳號
REMOTE_USER="ckhung"

# 2. kawas 伺服器的 IP 位址或網域
REMOTE_HOST="kawas.tiara.sinica.edu.tw"   # 例如: kawas.tiara.sinica.edu.tw

# 3. Python 腳本在 kawas 上的【絕對路徑】
# (請確認您已經把 Python 檔放在 kawas 上的這個位置)
REMOTE_PYTHON_SCRIPT="/theory/home/ckhung/SwinIR/recognition/trainningset_forsynline_generator.py" 

# 4. TIF 圖檔在 kawas 上生成後的【資料夾路徑】
# (對應您 Python 程式碼中的 output_dir，請寫絕對路徑)
REMOTE_OUTPUT_DIR="/theory/home/ckhung/raw_data/data_from_synline/" 

# 5. 您希望在本地端 (kolong) 存放 TIF 圖檔的資料夾名稱
LOCAL_OUTPUT_DIR="./raw_data/data_from_synline/"

# ==========================================
# 1. 觸發遠端執行 (SSH)
# ==========================================
echo ">>> [1/2] 正在連線至 kawas 並下達執行指令..."
echo " (可能需要輸入密碼。資料處理中請勿關閉視窗，請耐心等候...)"
echo "------------------------------------------------"

# 透過 SSH 遠端觸發 Python 腳本
ssh "${REMOTE_USER}@${REMOTE_HOST}" "module load python/3.10.2 && python3 ${REMOTE_PYTHON_SCRIPT}"

# 檢查 Python 執行是否成功
if [ $? -ne 0 ]; then
    echo "------------------------------------------------"
    echo "[ERROR] 遠端 Python 程式執行失敗，請檢查 kawas 上的程式碼或路徑設定。"
    exit 1
fi

echo "------------------------------------------------"
echo ">>> 遠端處理完成！"

# ==========================================
# 2. 透過 Rsync 抓取資料回本地端
# ==========================================
echo ">>> [2/2] 開始將生成的 .tif 檔案同步回本地端 (${LOCAL_OUTPUT_DIR})..."

# 確保本地端資料夾存在
mkdir -p "${LOCAL_OUTPUT_DIR}"

# 使用 rsync 進行同步 (-a 保持檔案屬性, -v 顯示進度, -P 支援斷點續傳)
# 注意：REMOTE_OUTPUT_DIR 後面的斜線很重要，代表同步資料夾內的檔案
rsync -avP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}" "${LOCAL_OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "================================================"
    echo ">>> 🎉 全部任務完成！您的 TIF 圖檔已安全下載至: ${LOCAL_OUTPUT_DIR}"
else
    echo "[ERROR] Rsync 同步失敗，請檢查網路連線。"
fi