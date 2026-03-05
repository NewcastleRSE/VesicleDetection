import pandas as pd
import shutil
import sys
from pathlib import Path

def process_good_ratings(csv_input_path):
    # 1. Resolve the absolute path of the CSV file
    csv_path = Path(csv_input_path).resolve()
    
    if not csv_path.exists():
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    # 2. Identify the base directory (where the CSV lives)
    # The masked_path column contains relative paths from this base directory.
    base_dir = csv_path.parent
    
    # 3. Create the 'good' folder inside that same base directory
    target_folder = base_dir / "good"
    target_folder.mkdir(exist_ok=True)
    print(f"Base Directory: {base_dir}")
    print(f"Target Folder: {target_folder}")

    # 4. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 5. Filter for 'Good' (case-insensitive and stripping whitespace)
    if 'rating' not in df.columns or 'masked_path' not in df.columns:
        print("Error: CSV must have 'rating' and 'masked_path' columns.")
        return

    good_rows = df[df['rating'].astype(str).str.strip().str.lower() == 'good']
    print(f"Found {len(good_rows)} rows with rating 'Good'. Starting copy...")

    # 6. Copy the files
    success_count = 0
    error_count = 0

    for _, row in good_rows.iterrows():
        relative_path_str = str(row['masked_path']).strip()
        
        # Combine base_dir + relative_path from CSV
        # This handles the "append" logic you requested
        full_source_path = (base_dir / relative_path_str).resolve()
        
        if not full_source_path.exists():
            print(f"File not found: {full_source_path}")
            error_count += 1
            continue
            
        try:
            # Get just the filename to create the destination path
            file_name = full_source_path.name
            dest_path = target_folder / file_name
            
            # Copy the file (shutil handles Path objects directly)
            shutil.copy2(full_source_path, dest_path)
            success_count += 1
        except Exception as e:
            print(f"Error copying {file_name}: {e}")
            error_count += 1

    print(f"\n--- Process Complete ---")
    print(f"Successfully copied: {success_count}")
    print(f"Errors/Missing: {error_count}")

if __name__ == "__main__":
    # Check if the user provided the CSV path as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python find_good_crops.py /path/to/your_file.csv")
    else:
        process_good_ratings(sys.argv[1])