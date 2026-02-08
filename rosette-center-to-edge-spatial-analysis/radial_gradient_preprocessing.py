import pandas as pd
import numpy as np
import glob
import argparse
from pathlib import Path
def parse_target_list(items):
    out = []
    for item in items:
        parts = [p.strip() for p in item.split(",")]
        out.extend([p for p in parts if p])
    return list(set(out))

def preprocess_multi_rosette_data(folder_path, master_metrics_filename, target_mz_list):
    directory = Path(folder_path)
    master_metrics_path = directory / master_metrics_filename
    output_file = directory / 'merged_multi_rosette_data_v2.csv'

    # 1. Load Master with Rounding
    cols_to_keep = ['X', 'Y', 'ncam only'] + (target_mz_list if target_mz_list else [])
    master_metrics = pd.read_csv(master_metrics_path, usecols=lambda x: x in cols_to_keep)
    master_metrics = master_metrics.rename(columns={'X': 'x', 'Y': 'y'})
    
    # FIX: Round to 2 decimal places to handle precision bugs
    master_metrics['x'] = master_metrics['x'].round(2)
    master_metrics['y'] = master_metrics['y'].round(2)

    # 2. Collect Rosette Files
    all_csvs = glob.glob(str(directory / "*.csv"))
    rosette_files = [f for f in all_csvs if Path(f).name != master_metrics_filename and "merged_multi" not in Path(f).name]
    
    # 3. Process & Compare
    print(f"{'Rosette ID':<15} | {'Initial Cells':<15} | {'Matched Cells':<15} | {'Lost'}")
    print("-" * 65)

    all_rosettes_list = []
    for f in rosette_files:
        temp_df = pd.read_csv(f)
        # Round coordinates here too
        temp_df['x'] = temp_df['x'].round(2)
        temp_df['y'] = temp_df['y'].round(2)
        
        rid = temp_df['rosette_id'].iloc[0] if not temp_df.empty else "Unknown"
        initial_count = len(temp_df)
        
        # Internal test merge for this specific file
        test_merge = pd.merge(temp_df, master_metrics, on=['x', 'y'], how='inner')
        matched_count = len(test_merge)
        lost = initial_count - matched_count
        
        print(f"{str(rid):<15} | {initial_count:<15} | {matched_count:<15} | {lost}")
        
        if matched_count > 0:
            all_rosettes_list.append(test_merge)

    if not all_rosettes_list:
        print("Error: No cells matched at all.")
        return

    final_df = pd.concat(all_rosettes_list, ignore_index=True)

    # 4. Final Lumen Labeling
    def label_lumen(group):
        threshold = group['distance'].quantile(0.20)
        group['cell_type'] = np.where(group['distance'] <= threshold, 'Lumen', 'Stroma')
        return group

    final_df = final_df.groupby('rosette_id', group_keys=False).apply(label_lumen)
    final_df.to_csv(output_file, index=False)
    print(f"\nFinal integrated file saved to {output_file.name}")

# ... (Argparse logic same as before)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="multiple_rosette_single_ROI_testing_data")
    parser.add_argument("--master", default="metrics region 1 ALL.csv")
    parser.add_argument("--target_mz", action="append", default=[])
    
    args = parser.parse_args()
    targets = parse_target_list(args.target_mz)
    
    preprocess_multi_rosette_data(args.folder, args.master, targets)