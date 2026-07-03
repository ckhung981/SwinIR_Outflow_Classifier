#!/bin/bash

# ==========================================
# System Configurations
# ==========================================
REMOTE_USER="ckhung"
REMOTE_HOST="kawas.tiara.sinica.edu.tw"

# 定義本地的 Python 腳本路徑 (假設與此 shell script 放同一個資料夾)
LOCAL_PYTHON_SCRIPT="./trainningset_forsynline_generator.py"

# 定義遠端 (kawas) 的暫存 Python 腳本路徑與輸出路徑
REMOTE_TEMP_SCRIPT="/theory/home/ckhung/temp_generator_$(date +%s).py" 
REMOTE_OUTPUT_DIR="/theory/home/ckhung/raw_data/data_multi_frame/"

LOCAL_OUTPUT_DIR="./raw_data/data_multi_frame/"

# ==========================================
# Target Data Parameters (Modify these directly)
# ==========================================
N_LIST="1 2 4 6"
M_LIST="6 30 60 90"
BP_LIST="1"
FRAME_LIST="00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023 00024 00025 00026 00027 00028 00029 00030"
INCLINATION_LIST="90"
SYNLINE_VERSION="88f988cb1622c04456df8444ca0f844c422998e2"

# Data Processing Parameters
BG_VALUE="1e-32"
RES_X="3.0"
RES_Y="3.0"
RES_Z="3.0"
TOLERANCE="0.20"
IGNORE_RES_LIMIT="1" # 1 means True, 0 means False

# ==========================================
# 0. Check Local Python Script
# ==========================================
if [ ! -f "$LOCAL_PYTHON_SCRIPT" ]; then
    echo "[ERROR] Local Python script not found at: $LOCAL_PYTHON_SCRIPT"
    exit 1
fi

# ==========================================
# 1. Upload Python Script to Remote (SCP)
# ==========================================
echo ">>> [1/3] Uploading local Python script to kawas..."
scp "$LOCAL_PYTHON_SCRIPT" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TEMP_SCRIPT}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upload Python script to kawas."
    exit 1
fi

# ==========================================
# 2. Trigger Remote Execution (SSH)
# ==========================================
echo ">>> [2/3] Executing script on kawas..."
echo " (Please wait patiently without closing the window...)"
echo "------------------------------------------------"

# Construct the python command with arguments using the TEMP script
PYTHON_CMD="python3 ${REMOTE_TEMP_SCRIPT} \
    --n_list ${N_LIST} \
    --m_list ${M_LIST} \
    --bp_list ${BP_LIST} \
    --frame_list ${FRAME_LIST} \
    --inclinations ${INCLINATION_LIST} \
    --synline_version ${SYNLINE_VERSION} \
    --bg_value ${BG_VALUE} \
    --res_x ${RES_X} \
    --res_y ${RES_Y} \
    --res_z ${RES_Z} \
    --tolerance ${TOLERANCE} \
    --ignore_res_limit ${IGNORE_RES_LIMIT} \
    --output_dir ${REMOTE_OUTPUT_DIR}"

# Execute via SSH and immediately delete the temp script afterwards (cleanup)
ssh "${REMOTE_USER}@${REMOTE_HOST}" "module load python/3.10.2 && ${PYTHON_CMD}; rm -f ${REMOTE_TEMP_SCRIPT}"

if [ $? -ne 0 ]; then
    echo "------------------------------------------------"
    echo "[ERROR] Remote Python execution failed."
    exit 1
fi

echo "------------------------------------------------"
echo ">>> Remote execution completed successfully."

# ==========================================
# 3. Sync Data to Local Directory
# ==========================================
echo ">>> [3/3] Syncing generated .tif files to local directory (${LOCAL_OUTPUT_DIR})..."

mkdir -p "${LOCAL_OUTPUT_DIR}"

rsync -avP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}" "${LOCAL_OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "================================================"
    echo ">>> All tasks completed. TIF files downloaded to: ${LOCAL_OUTPUT_DIR}"
else
    echo "[ERROR] Rsync failed. Please check your network connection."
fi