'''
This script processes Synline simulation data to generate training images 
for a SwinIR-based recognition model. It performs the following steps:
1. Configures parameters for data processing strictly from command line arguments.
2. Searches for Synline data folders matching specific criteria.
3. Validates the resolution of candidate folders against target values with a tolerance.
4. Reads the relevant binary data file, reshapes it, applies necessary rotations and flips.
5. Converts the data to logarithmic scale and saves it as a TIF image.
6. Logs any cases that fail due to missing files, resolution mismatches, or processing errors.
'''

import os
import glob
import re
import numpy as np
import tifffile
import argparse

AU_CM = 1.5e13

def get_resolution(folder_path, axis_suffix):
    """
    Find the coordinate file with prefix '3d_' and calculate its resolution.
    """
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
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Generate training images from Synline simulation data.")
    parser.add_argument('--n_list', type=int, nargs='+', default=[6], help="List of N values")
    parser.add_argument('--m_list', type=int, nargs='+', default=[90], help="List of m values")
    parser.add_argument('--bp_list', type=float, nargs='+', default=[1.0], help="List of bp values")
    parser.add_argument('--frame_list', type=str, nargs='+', default=["00030"], help="List of frame strings")
    parser.add_argument('--inclinations', type=int, nargs='+', default=[90, 105, 120, 135, 150, 165], help="List of inclinations")
    parser.add_argument('--synline_version', type=str, default='88f988cb1622c04456df8444ca0f844c422998e2', help="Synline version hash")
    
    parser.add_argument('--bg_value', type=float, default=1e-32, help="Background value replacement")
    parser.add_argument('--res_x', type=float, default=3.0, help="Target resolution for X axis")
    parser.add_argument('--res_y', type=float, default=3.0, help="Target resolution for Y axis")
    parser.add_argument('--res_z', type=float, default=3.0, help="Target resolution for Z axis")
    parser.add_argument('--tolerance', type=float, default=0.20, help="Allowed resolution tolerance")
    parser.add_argument('--ignore_res_limit', type=int, choices=[0, 1], default=1, help="1 to ignore resolution limits, 0 to enforce")
    parser.add_argument('--output_dir', type=str, default='./raw_data/data_from_synline', help="Output directory for generated TIF files")
    
    args = parser.parse_args()

   
    THEORY_PREFIX = f"/theory/lts/ckhung/"
    SCRATCH_PREFIX = f"/scratch/data/ckhung/"

    N_list = args.n_list
    m_list = args.m_list      
    bp_list = args.bp_list
    frame_list = args.frame_list 
    inclinations = args.inclinations
    synline_version = args.synline_version
    
    background_value = args.bg_value
    target_resolutions = {
        'x': args.res_x,  
        'y': args.res_y,  
        'z': args.res_z   
    }
    allowed_tolerance = args.tolerance
    ignore_res_limit = bool(args.ignore_res_limit)
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    failed_cases = []
    ignored_res_cases = []

    # --- Loop Processing ---
    for bp in bp_list:
        if bp == 0:
            bp_str = '0bp'
        elif bp == 0.1:
            bp_str = '0p1bp'
        elif bp == 1:
            bp_str = '1bp'
        else:
            if bp.is_integer():
                bp_str = f'{int(bp)}bp'
            else:
                bp_str = f'{bp}bp'

        for n in N_list:
            for m in m_list:
                for inclination in inclinations: 
                    for frame in frame_list:
                        case_name = f"n={n}, m={m}, bp={bp_str}, inclination={inclination}, frame={frame}"
                        print("=========================================")
                        print(f"Processing: {case_name}")
                        
                        if inclination == 90:
                            if m == 999:
                                search_pattern = (
                                    f"{THEORY_PREFIX}/Synline_datacubes/datacube_vtheta_0v2/"
                                    f"n{n}minf_0v2_{bp_str}_csw6e4_vw100_*/"
                                    f"ly_synline_v0.4.0/small512_16/view90w*h*s*{frame}*" 
                                )
                            else:
                                search_pattern = (
                                    f"{SCRATCH_PREFIX}/Synline_data/"
                                    f"n{n}m{m}_{bp_str}_csw6e4_vw100_*/"
                                    f"ly_synline_v0.4.0/{synline_version}/view90w*h*s*{frame}*" 
                                )
                        else:
                            if m == 999:
                                search_pattern = (
                                    f"{THEORY_PREFIX}/Synline_datacubes/datacube_vtheta_0v2/"
                                    f"n{n}minf_0v2_{bp_str}_csw6e4_vw100_*/"
                                    f"ly_synline_v0.4.0/small512_16/view{inclination}w*h*s*{frame}*"
                                )
                            else:
                                search_pattern = (
                                    f"{SCRATCH_PREFIX}/Synline_data/"
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
                        
                        valid_base_folder = None
                        fallback_folder = None
                        fallback_res = None
                        
                        for folder in matched_paths:
                            print(f"Checking candidate folder: {os.path.basename(folder)}")
                            
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
                        
                        if inclination == 90:
                            match = re.search(r'w(\d+)h(\d+)', base_folder) 
                        else:
                            match = re.search(r'w(\d+)h(\d+)s', base_folder)
                            
                        if not match:
                            error_msg = f"Failed to extract dimension information from folder name: {base_folder}"
                            print(error_msg)
                            failed_cases.append({'case': case_name, 'reason': error_msg})
                            continue
                        
                        folder_x_num = int(match.group(1)) 
                        folder_y_num = int(match.group(2)) 
                        
                        print(f"Target folder located: {base_folder}")
                        print(f"Target data file located: {os.path.basename(full_path_data)}")
                        print(f"Dimensions parsed from folder name: w={folder_x_num}, h={folder_y_num}")

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
                        
                        x_num = os.path.getsize(coord_x_file) // 8
                        y_num = os.path.getsize(coord_y_file) // 8
                        print(f"Actual image dimensions redefined from coord files: x_num(w)={x_num}, y_num(h)={y_num}")

                        try:
                            expected_elements = x_num * y_num
                            flat_data = np.fromfile(full_path_data, dtype=np.float64, count=expected_elements)
                            
                            if len(flat_data) < expected_elements:
                                raise ValueError(f"Data file too small! Expected {expected_elements}, got {len(flat_data)}")
                            
                            raw_image = flat_data.reshape((x_num, y_num))
                            rotated_image = np.rot90(raw_image, 1)
                            flipped_image = np.fliplr(rotated_image)
                            final_image = np.concatenate((flipped_image, rotated_image), axis=1)
                            
                        except Exception as e:
                            error_msg = f"Error reading or processing binary file: {e}"
                            print(error_msg)
                            failed_cases.append({'case': case_name, 'reason': error_msg})
                            continue
                        
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

    print("\n" + "="*50)
    print("Processing Summary")
    print("="*50)
    
    if ignored_res_cases:
        print(f"[!] {len(ignored_res_cases)} case(s) were processed ignoring resolution limits.\n")
        print("Ignored Resolution Cases Detail:")
        for idx, info in enumerate(ignored_res_cases, 1):
            print(f"[{idx}] {info['case']}")
            print(f"    Folder: {info['folder']}")
            print(f"    Questionable Resolutions -> {info['resolutions']}\n")
        print("-" * 50)
        
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