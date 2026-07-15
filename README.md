# Activate environment
source ~/swin_transformer_env/bin/activate
# Download Data
1. sh trainningset_download_data.sh

# Preprocessing
2. python trainningset_normalization.py
3. python trainningset_covolution.py
4. python trainningset_cropper.py
5. python trainningset_extract_testcase.py

# Trainning
6. (optional) tmux
7. python main_train_recon.py

* "tmux attach" to rusume the session

After trainning, better to make a new directory and move data/, model_weight/, normalized_data/, resized_conv_data/ into this new directory