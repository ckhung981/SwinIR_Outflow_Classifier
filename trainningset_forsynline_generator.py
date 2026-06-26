'''
This script processes Synline simulation data to generate training images 
for a SwinIR-based recognition model. It performs the following steps:
1. Configures parameters for data processing and model training.
2. Searches for Synline data folders matching specific criteria (N, m, bp, inclination).
3. Validates the resolution of candidate folders against target values with a tolerance.
4. Reads the relevant binary data file, reshapes it, applies necessary rotations and flips.
5. Converts the data to logarithmic scale and saves it as a TIF image.
6. Logs any cases that fail due to missing files, resolution mismatches, or processing errors
'''

import os
import glob
import re
import numpy as np
import tifffile

AU_CM = 1.5e13

def get_resolution(folder_path, axis_suffix):
    """
    Find the coordinate file with prefix '3d_' and calculate its resolution.
    """
    # Enforce the '3d_' prefix in the file search pattern
    # Using wildcard before suffix to catch both '3d_x1b' and '3d_x1b_part2'
    search_path = os.path.join(folder_path, f'3d_*{axis_suffix}*')
    matched_files = glob.glob(search_path)
    
    if not matched_files:
        return None
        
    try:
        coord_file = matched_files[0]
        coord_data = np.fromfile(coord_file, dtype=np.float64)
        
        if len(coord_data) < 2:
            return 0.0
            
        resolution = (np.max(coord_data) - np.min(coord_data)) / AU_CM / len(coord_data)
        return resolution
    except Exception as e:
        print(f"Error reading coordinate file {coord_file}: {e}")
        return None

def check_tolerance(actual_res, target_res, tolerance=0.05):
    """
    Check if the actual resolution is within the tolerance of the target resolution.
    """
    if actual_res is None or target_res is None:
        return False
    if actual_res == 0.0:
        return True 
        
    error = abs(actual_res - target_res) / target_res
    return error <= tolerance

def main():
    # --- [新增] 本地掛載點前綴設定 ---
    # 取得當前使用者的 Home 目錄路徑 (展開 '~')
    home_dir = os.path.expanduser("~")
    # 定義與 shell script 一致的掛載點
    THEORY_PREFIX = f"{home_dir}/mnt/kawas_theory"
    SCRATCH_PREFIX = f"{home_dir}/mnt/kawas_scratch"
    # ----------------------------------

    # --- 1. Parameters Configuration ---
    #N_list = [1, 2, 4, 6]
    N_list = [6]
    m_list = [90]      
    bp_list = [1]
    frame = "00030"
    inclinations = [90, 105, 120, 135, 150, 165]     
    background_value = 1e-32
    synline_version = '88f988cb1622c04456df8444ca0f844c422998e2'
    # Target resolutions for x, y, z axes 
    target_resolutions = {
        'x': 3.0,  
        'y': 3.0,  
        'z': 3.0   
    }
    allowed_tolerance = 0.20
    
    # Ignore resolution limits if no perfect match is found
    ignore_res_limit = True 
    
    output_dir = './raw_data/data_from_synline'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to keep track of failed cases and ignored limit cases
    failed_cases = []
    ignored_res_cases = []

    # --- 2. Loop Processing ---
    for bp in bp_list:
        if bp == 0:
            bp_str = '0bp'
        elif bp == 0.1:
            bp_str = '0p1bp'
        elif bp == 1:
            bp_str = '1bp'
        else:
            bp_str = f'{bp}bp'

        for n in N_list:
            for m in m_list:
                for inclination in inclinations: 
                    case_name = f"n={n}, m={m}, bp={bp_str}, inclination={inclination}"
                    print("=========================================")
                    print(f"Processing: {case_name}")
                    
                    # --- 3. Build Search Pattern (Modified for Local Mounts) ---
                    if inclination == 90:
                        if m == 999:
                            search_pattern = (
                                f"{THEORY_PREFIX}/lts/ckhung/Synline_datacubes/datacube_vtheta_0v2/"
                                f"n{n}minf_0v2_{bp_str}_csw6e4_vw100_*/"
                                f"ly_synline_v0.4.0/small512_16/view90w*h*s*{frame}*" 
                            )
                        else:
                            search_pattern = (
                                f"{SCRATCH_PREFIX}/ckhung/Synline_data/"
                                f"n{n}m{m}_{bp_str}_csw6e4_vw100_*/"
                                f"ly_synline_v0.4.0/{synline_version}/view90w*h*s*{frame}*" 
                            )
                    else:
                        if m == 999:
                            search_pattern = (
                                f"{THEORY_PREFIX}/lts/ckhung/Synline_datacubes/datacube_vtheta_0v2/"
                                f"n{n}minf_0v2_{bp_str}_csw6e4_vw100_*/"
                                f"ly_synline_v0.4.0/small512_16/view{inclination}w*h*s*{frame}*"
                            )
                        else:
                            search_pattern = (
                                f"{SCRATCH_PREFIX}/ckhung/Synline_data/"
                                f"n{n}m{m}_{bp_str}_csw6e4_vw100_*/"
                                f"ly_synline_v0.4.0/{synline_version}/view{inclination}w*h*s*{frame}*"
                            )

                    print(f"Searching directory pattern: {search_pattern}")
                    
                    matched_paths = glob.glob(search_pattern)
                    
                    if not matched_paths:
                        error_msg = "Directory not found matching pattern."
                        print(f"{error_msg} Skipping.")
                        failed_cases.append({'case': case_name, 'reason': error_msg})
                        continue
                    
                    # --- 4. Find the folder that matches the target resolutions ---
                    valid_base_folder = None
                    fallback_folder = None
                    fallback_res = None
                    
                    for folder in matched_paths:
                        print(f"Checking candidate folder: {os.path.basename(folder)}")
                        
                        # get_resolution wildcard '3d_*x1b*' natively handles both '3d_x1b' and '3d_x1b_part2'
                        res_x = get_resolution(folder, 'x1b')
                        res_y = get_resolution(folder, 'x2b')
                        res_z = get_resolution(folder, 'x3b')
                        
                        print(f"  Calculated Resolutions -> X: {res_x}, Y: {res_y}, Z: {res_z}")
                        
                        match_x = check_tolerance(res_x, target_resolutions['x'], allowed_tolerance)
                        match_y = check_tolerance(res_y, target_resolutions['y'], allowed_tolerance)
                        
                        if res_z is not None:
                            match_z = check_tolerance(res_z, target_resolutions['z'], allowed_tolerance)
                        else:
                            match_z = True
                        
                        if match_x and match_y and match_z:
                            valid_base_folder = folder
                            print("  Match found. Resolution error is within tolerance.")
                            break
                        else:
                            print("  Resolution error exceeds tolerance.")
                            if fallback_folder is None:
                                fallback_folder = folder
                                fallback_res = (res_x, res_y, res_z)
                            
                    # Handle the case where no folder perfectly matches
                    if valid_base_folder is None:
                        if ignore_res_limit and fallback_folder is not None:
                            valid_base_folder = fallback_folder
                            res_warning = f"X: {fallback_res[0]}, Y: {fallback_res[1]}, Z: {fallback_res[2]}"
                            print(f"  [WARNING] No folder matched the target resolutions. 'ignore_res_limit' is True.")
                            print(f"  [WARNING] Proceeding with fallback folder: {os.path.basename(valid_base_folder)}")
                            print(f"  [WARNING] Questionable Resolutions -> {res_warning}")
                            
                            ignored_res_cases.append({
                                'case': case_name, 
                                'folder': os.path.basename(valid_base_folder),
                                'resolutions': res_warning
                            })
                        else:
                            error_msg = "No folder matched the target resolution criteria and ignore flag is False."
                            print(f"{error_msg} Skipping.")
                            failed_cases.append({'case': case_name, 'reason': error_msg})
                            continue
                    
                    base_folder = valid_base_folder
                    
                    # --- 5. Find the data file containing column_density ---
                    if inclination == 90:
                        data_file_pattern = os.path.join(base_folder, 'part1_column_density')
                    else:
                        data_file_pattern = os.path.join(base_folder, 'part*_column_density')
                        
                    print(f"Searching for data file: {data_file_pattern}")
                    
                    matched_data_files = glob.glob(data_file_pattern)
                    
                    if not matched_data_files:
                        error_msg = f"File matching pattern '{os.path.basename(data_file_pattern)}' not found in {base_folder}."
                        print(f"{error_msg} Skipping.")
                        failed_cases.append({'case': case_name, 'reason': error_msg})
                        continue
                        
                    full_path_data = matched_data_files[0] 
                    
                    # --- 6. Extract dimensions (w & h) from folder name (Regex) ---
                    if inclination == 90:
                        match = re.search(r'w(\d+)h(\d+)', base_folder) 
                    else:
                        match = re.search(r'w(\d+)h(\d+)s', base_folder)
                        
                    if not match:
                        error_msg = f"Failed to extract dimension information from folder name: {base_folder}"
                        print(error_msg)
                        failed_cases.append({'case': case_name, 'reason': error_msg})
                        continue
                    
                    # Original logic from folder name
                    folder_x_num = int(match.group(1)) 
                    folder_y_num = int(match.group(2)) 
                    
                    print(f"Target folder located: {base_folder}")
                    print(f"Target data file located: {os.path.basename(full_path_data)}")
                    print(f"Dimensions parsed from folder name: w={folder_x_num}, h={folder_y_num}")

                    # --- NEW 6.5: Redefine dimensions from coordinate files ---
                    if inclination == 90:
                        coord_x_file = os.path.join(base_folder, '3d_x1b')
                        coord_y_file = os.path.join(base_folder, '3d_x2b')
                    else:
                        coord_x_file = os.path.join(base_folder, '3d_x1b_part2')
                        coord_y_file = os.path.join(base_folder, '3d_x2b_part2')
                    
                    if not (os.path.exists(coord_x_file) and os.path.exists(coord_y_file)):
                        error_msg = f"Coordinate files missing in {base_folder} ({os.path.basename(coord_x_file)} or {os.path.basename(coord_y_file)}). Cannot redefine grid size."
                        print(error_msg)
                        failed_cases.append({'case': case_name, 'reason': error_msg})
                        continue
                    
                    # Calculate dimensions by checking file size in bytes and dividing by 8 (sizeof double)
                    x_num = os.path.getsize(coord_x_file) // 8
                    y_num = os.path.getsize(coord_y_file) // 8
                    print(f"Actual image dimensions redefined from coord files: x_num(w)={x_num}, y_num(h)={y_num}")

                    # --- 7. Read binary data, reshape, rotate, and mirror ---
                    try:
                        expected_elements = x_num * y_num
                        # Add 'count' to only read exactly x_num * y_num elements, ignoring MPI padding
                        flat_data = np.fromfile(full_path_data, dtype=np.float64, count=expected_elements)
                        
                        if len(flat_data) < expected_elements:
                            raise ValueError(f"Data file too small! Expected {expected_elements}, got {len(flat_data)}")
                        
                        # Ensure order='F' is used if MATLAB generated binary files
                        raw_image = flat_data.reshape((x_num, y_num))
                        
                        # Rotate 90 degrees counter-clockwise
                        rotated_image = np.rot90(raw_image, 1)
                        
                        # Mirror copy across y-axis (from positive x to negative x)
                        flipped_image = np.fliplr(rotated_image)
                        final_image = np.concatenate((flipped_image, rotated_image), axis=1)
                        
                    except Exception as e:
                        error_msg = f"Error reading or processing binary file: {e}"
                        print(error_msg)
                        failed_cases.append({'case': case_name, 'reason': error_msg})
                        continue
                    
                    # --- 8. Value conversion and TIF saving ---
                    final_image[final_image <= 0.0] = background_value
                    log_image = np.log10(final_image).astype(np.float32) 
                    
                    output_filename = f'output_n{n}_m{m}_{bp_str}_{inclination}deg_{frame}_raw.tif'
                    output_filepath = os.path.join(output_dir, output_filename)
                    
                    try:
                        tifffile.imwrite(output_filepath, log_image)
                        print(f"Successfully saved TIF: {output_filename}\n")
                    except Exception as e:
                        error_msg = f"Error saving TIF file: {e}"
                        print(error_msg)
                        failed_cases.append({'case': case_name, 'reason': error_msg})

    # --- 9. Print Final Summary ---
    print("\n" + "="*50)
    print("Processing Summary")
    print("="*50)
    
    # Print ignored resolution cases first
    if ignored_res_cases:
        print(f"[!] {len(ignored_res_cases)} case(s) were processed ignoring resolution limits.\n")
        print("Ignored Resolution Cases Detail:")
        for idx, info in enumerate(ignored_res_cases, 1):
            print(f"[{idx}] {info['case']}")
            print(f"    Folder: {info['folder']}")
            print(f"    Questionable Resolutions -> {info['resolutions']}\n")
        print("-" * 50)
        
    # Print failed cases
    if not failed_cases:
        if ignored_res_cases:
            print("All other cases processed successfully!")
        else:
            print("All cases processed successfully!")
    else:
        print(f"[X] Warning: {len(failed_cases)} case(s) failed during processing.\n")
        print("Failed Cases Detail:")
        for idx, fail_info in enumerate(failed_cases, 1):
            print(f"[{idx}] {fail_info['case']}")
            print(f"    Reason: {fail_info['reason']}\n")
    print("="*50)

if __name__ == '__main__':
    main()
