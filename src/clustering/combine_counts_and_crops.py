from os import path
import shutil
import sys
import pandas as pd
import os
import glob
import re
from find_good_crops import process_good_ratings


def find_and_combine(path='.'):
    """ This function will find all the unique prefixes from the labels files in the specified path, 
    and for each prefix, it will find the relevant folders and files, it will rename the internal 
    files to have the prefix appended if it isn't already, and then it will move the files into a 
    combined folder to hold all the processed crops if they are not already there. 
    It will also create a combined csv file with the counts for each cluster and the new filenames. 
    The counts will be combined according to the "clusters to combine" column in the ratings file, 
    and the new filenames will be in the format "{prefix}_cluster_{id}_masked.h5".

    Args:
        path (str, optional): The path to the directory containing the cropped_convexhull folder and labels csv file, 
                              as produced by the cluster_crop_pipeline. Defaults to the current working directory.
    """
    # --- Execution Loop ---
    all_results = []
    # Find all unique prefixes from your labels files
    prefixes = [os.path.basename(f).replace('labels.csv', '') for f in glob.glob(f"{path}/*labels.csv")]
    print(f"Found {len(prefixes)} unique prefixes: {prefixes}")
    if len(prefixes) == 0:
        print("No labels files found. Please ensure there are files ending with 'labels.csv' in the specified path.")
        return
    elif len(prefixes) == 1:
        print(f"Only one prefix found: {prefixes[0]}. No combination needed, but processing will still occur.")
        process_and_combine_crops(prefixes[0], path, combine=False)
    else:
        print(f"Multiple label files found. Processing and combining crops for all prefixes.")
        for p in prefixes:
            print(f"Processing {p}...")
            process_and_combine_crops(p,path,combine=True)


def process_and_combine_crops(prefix, base_path='.', combine=True):
    """This function will find folders with the prefix before convexhullmask or vesiclemask, it will rename the internal files to have the prefix appended if it isn't already,
    and then it will move the files into a combined folder to hold all the processed crops if they are not already there. It will also create a combined csv file with the counts for each cluster and the new filenames.
    The counts will be combined according to the "clusters to combine" column in the ratings file, and the new filenames will be in the format "{prefix}_cluster_{id}_masked.h5".

    Args:
        prefix (string): An identifier for the dataset that is used to find the relevant folders and files. For example, if prefix is "19-13_subvolume_0647-1670", it will look for folders like "19-13_subvolume_0647-1670_convexhullmask" and "19-13_subvolume_0647-1670_vesiclemask", 
                         and it will look for files like "cluster_4_masked.h5" inside those folders. It will then create new files like "19-13_subvolume_0647-1670_cluster_4_masked.h5" in a combined folder, and it will create a csv file with the counts for each cluster and the new filenames.
    """
    # find the relevant folders and file
    convex_folder = glob.glob(f"{base_path}/{prefix}*cropped_convexhull*")
    if not convex_folder:
        print(f"No convex hull folder found for prefix {prefix} in {base_path}.")
        convex_folder = None
    else:        
        convex_folder = convex_folder[0]
    vesicle_folder = glob.glob(f"{base_path}/{prefix}*cropped_vesicle*")
    if not vesicle_folder:
        print(f"No vesicle folder found for prefix {prefix} in {base_path}.")
        vesicle_folder = None
    else:
        vesicle_folder = vesicle_folder[0]
    labels_file = f"{base_path}/{prefix}labels.csv"
    if not os.path.exists(labels_file):
        print(f"Labels file {labels_file} not found for prefix {prefix}. Skipping.")
        return
    if not convex_folder and not vesicle_folder:
        print(f"No folders found for prefix {prefix} in {base_path}. Skipping.")
        return
    
    if combine:
        combined_folder = f"{base_path}/combined_data"
        os.makedirs(combined_folder, exist_ok=True)
        convex_combined = "convexhull_combined"
        vesicle_combined = "vesicle_combined"
        os.makedirs(os.path.join(combined_folder, convex_combined), exist_ok=True)
        os.makedirs(os.path.join(combined_folder, vesicle_combined), exist_ok=True)
    else:
        combined_folder = base_path
        # make the labels combined csv file if it doesn't exist, with the appropriate header:
    labels_combined_path = os.path.join(combined_folder, f"{prefix}_labelscombined.csv")
    if not os.path.exists(labels_combined_path):

        with open(labels_combined_path, "w", encoding="utf-8") as f:
            f.write("cluster,positive_count,negative_count,cluster_size\n")
   
    ratings_path = glob.glob(f"{convex_folder}/*ratings.csv") if convex_folder else None
    print(f"Looking for ratings file in {convex_folder} for prefix {prefix}. Found: {ratings_path}")
    if ratings_path:
        # check if the ratings have been completed by looking for any 'Good' ratings in the ratings file, and if there are any, then combine the counts according to the "clusters to combine" column in the ratings file. If there are no 'Good' ratings, then just use the original counts without combination.
        df_ratings = pd.read_csv(ratings_path[0])
        if 'rating' in df_ratings.columns and 'masked_path' in df_ratings.columns:
            if df_ratings['rating'].astype(str).str.strip().str.lower().eq('good').any():
                print(f"Ratings file found for prefix {prefix}. Combining counts according to ratings.")
                counts_map = get_combined_counts_map(labels_file, ratings_path[0])
                # copy the ratings file into the vesicle folder if it exists, since the process_files function will look for the good folder in the same folder as the ratings file. If there is no vesicle folder, then just keep the ratings file in the convex hull folder.
                if vesicle_folder and not os.path.exists(f"{vesicle_folder}/{ratings_path[0]}"):
                    shutil.copy(ratings_path[0], vesicle_folder)
                if os.path.exists(f"{convex_folder}/good"):
                    print(f"'Good' folder already exists in {convex_folder}. Skipping creation of 'good' folder.")
                    # check if good exists in the vesicle folder as well, and if not, copy the good folder from the convex hull folder to the vesicle folder if it exists, since the process_files function will look for the good folder in the same folder as the ratings file. If there is no vesicle folder, then create it with the process_good_ratings function.
                    if vesicle_folder and not os.path.exists(f"{vesicle_folder}/good"):
                        print(f"'Good' folder not found in {vesicle_folder}. Creating good version for vesicles too")
                        ratings_path_in_vesicle = glob.glob(f"{vesicle_folder}/*ratings.csv")
                        if not os.path.exists(ratings_path_in_vesicle[0]):
                            shutil.copy(ratings_path[0], vesicle_folder)
                        process_good_ratings(ratings_path_in_vesicle[0])
                else: # there are good ratings but no good folder
                    print(f"'Good' ratings found in ratings file for prefix {prefix}. Creating 'good' folder and copying relevant files.")
                    process_good_ratings(ratings_path[0])
                    if vesicle_folder and not os.path.exists(f"{vesicle_folder}/good"):
                        ratings_path_in_vesicle = glob.glob(f"{vesicle_folder}/*ratings.csv")
                        if not os.path.exists(ratings_path_in_vesicle[0]):
                            shutil.copy(ratings_path[0], vesicle_folder)
                        print(f"'Good' folder not found in {vesicle_folder}. Creating good version for vesicles too")
                        process_good_ratings(ratings_path_in_vesicle[0])
            else:
                print(f"No 'Good' ratings found in ratings file for prefix {prefix}. Do you need to complete the ratings for this dataset? Using original counts without combination or filtering by ratings.")
                counts_map = get_combined_counts_map(labels_file, "")

    else:
        print(f"No ratings file found for prefix {prefix}. Using original counts without combination or filtering by ratings.")
        counts_map = get_combined_counts_map(labels_file, "")

    # find the relevant files and rename and move them if necessary, and add the filename to the combined csv file with the counts
    if combine:
        if convex_folder:
            process_files(convex_folder, os.path.join(combined_folder, convex_combined), counts_map, labels_combined_path, prefix, combined=combine)
        if vesicle_folder:
            process_files(vesicle_folder, os.path.join(combined_folder, vesicle_combined), counts_map, labels_combined_path, prefix, combined=combine)
    else:
        if convex_folder:
            process_files(convex_folder, convex_folder, counts_map, labels_combined_path, prefix, combined=combine)
        if vesicle_folder:
            process_files(vesicle_folder, vesicle_folder, counts_map, labels_combined_path, prefix, combined=combine)



def get_combined_counts_map(labels_file, ratings_path):
    """This function will load the labels file to get the original counts for each cluster,
      and if a ratings file is provided and contains 'Good' ratings, it will combine the counts 
      according to the "clusters to combine" column in the ratings file. It will return a dictionary
      mapping each cluster ID (or combined cluster ID) to its positive count, negative count, and cluster size.
      
      Args:
          labels_file (str): The path to the labels CSV file that contains the original counts for each cluster.
          ratings_path (str): The path to the ratings CSV file that contains the ratings and the "clusters to combine" 
                              information. If this is an empty string, then no combination will be done and the 
                              original counts from the labels file will be used.
      
      """
    print(f"Loading labels from {labels_file}...")
    df_labels = pd.read_csv(labels_file)
    # Clean the labels to have a string integer index
    df_labels['cluster_id'] = df_labels['cluster'].astype(str).str.replace('cluster_', '').str.replace('_masked.h5', '')
    df_labels = df_labels.set_index('cluster_id')

    # Default mapping: every cluster maps to its own original counts
    counts_map = {}
    for cid, row in df_labels.iterrows():
        counts_map[cid] = {
            'pos': row['positive_count'],
            'neg': row['negative_count'],
            'size': row['cluster_size']
        }

    if os.path.exists(ratings_path):
        df_ratings = pd.read_csv(ratings_path)
        df_ratings.columns = df_ratings.columns.str.strip()

        if 'clusters to combine' not in df_ratings.columns:
            print(f"Warning: 'clusters to combine' column not found in ratings file {ratings_path}. Skipping combination for this file.")
            return counts_map
        
        for _, row in df_ratings.iterrows():
            target_id = str(row['id'])
            # Split by semicolon as per your data
            to_combine = str(row['clusters to combine']).split(';')

            # add sanity checks in case of human error - the to_combine list should only contain cluster IDs that are present in the labels file, and if any are not, print a warning and skip them. Also, if the target_id is not in the labels file, print a warning and skip it.
            if target_id not in df_labels.index:
                print(f"Warning: Target ID {target_id} not found in labels file. Skipping this combination and just using the original count.")
                counts_map[target_id] = {
                    'pos': df_labels.loc[target_id, 'positive_count'] if target_id in df_labels.index else 0,
                    'neg': df_labels.loc[target_id, 'negative_count'] if target_id in df_labels.index else 0,
                    'size': df_labels.loc[target_id, 'cluster_size'] if target_id in df_labels.index else 0
                }
                continue
            valid_cluster_ids = [cid.strip() for cid in to_combine if cid.strip() in df_labels.index]
            invalid_cluster_ids = [cid.strip() for cid in to_combine if cid.strip() not in df_labels.index]
            if invalid_cluster_ids:
                print(f"Warning: The following cluster IDs to combine for target ID {target_id} were not found in labels file and will be skipped: {invalid_cluster_ids}")
            # check that the valid cluster ids list is not empty after removing invalid cluster ids, and if it is empty, print a warning and skip the combination for this target_id
            if not valid_cluster_ids:
                print(f"Warning: No valid cluster IDs to combine for target ID {target_id} after removing invalid cluster IDs. Skipping this combination and just using the original count.")
                counts_map[target_id] = {
                    'pos': df_labels.loc[target_id, 'positive_count'] if target_id in df_labels.index else 0,
                    'neg': df_labels.loc[target_id, 'negative_count'] if target_id in df_labels.index else 0,
                    'size': df_labels.loc[target_id, 'cluster_size'] if target_id in df_labels.index else 0
                }
                continue
            # check whether the target_id is also in the to_combine list, if it isn't then add it to the list so that its counts are included in the sum, and if it is, then we don't need to do anything since its counts will already be included in the sum when we loop through the to_combine list
            if target_id not in valid_cluster_ids:
                print(f"Note: Target ID {target_id} not found in 'clusters to combine' list for this target ID. Adding it to the list to include its counts in the combination.")
                valid_cluster_ids.append(target_id)
            
            # Check for any duplicate cluster IDs in the to_combine list, and if there are any, print a warning and remove the duplicates before summing the counts, since duplicate cluster IDs would lead to double counting of those clusters in the combined counts.
            if len(valid_cluster_ids) != len(set(valid_cluster_ids)):
                print(f"Warning: Duplicate cluster IDs found in 'clusters to combine' for target ID {target_id}. Removing duplicates to avoid double counting. Duplicates: {[cid for cid in valid_cluster_ids if valid_cluster_ids.count(cid) > 1]}")
                valid_cluster_ids = list(set(valid_cluster_ids))

            pos_sum, neg_sum, size_sum = 0, 0, 0
            for cid in valid_cluster_ids:
                cid = cid.strip()
                if cid in df_labels.index:
                    pos_sum += df_labels.loc[cid, 'positive_count']
                    neg_sum += df_labels.loc[cid, 'negative_count']
                    size_sum += df_labels.loc[cid, 'cluster_size']
            
            # Override the map with the summed values
            counts_map[target_id] = {'pos': pos_sum, 'neg': neg_sum, 'size': size_sum}
            
    return counts_map

def process_files(folder, combined_folder, counts_map, labels_combined_path, prefix, combined=True):
    """This function will look for files in the specified folder that match the pattern "*cluster_{id}_masked.h5",
      and for each file, it will check if the cluster ID is in the counts_map. 
      If it is, it will rename the file to have the prefix appended if it doesn't already, 
      and then it will copy the file to the combined_folder if combining, or rename it in place if not combining.
      It will also add an entry to the labels_combined_path CSV file with the new filename and the corresponding counts from the counts_map.

    Args:
        folder (str): The path to the folder containing the masked files.
        combined_folder (str): The path to the folder where combined files will be stored.
        counts_map (dict): A dictionary mapping each cluster ID to its combined counts.
        labels_combined_path (str): The path to the CSV file where combined labels will be stored.
        prefix (str): The prefix to be added to the filenames of the combined files.
        combined (bool, optional): Whether to combine the files. Defaults to True.
    """

    if not os.path.exists(f"{folder}/good"):
        print(f"Good subfolder not found in {folder}. Using original masked files without filtering by ratings.")
        crop_dir = f"{folder}/masked"
        if not os.path.exists(f"{folder}/masked"):
            print(f"Masked subfolder not found in {folder}. Skipping processing for this folder.")
            return
    else:
        print(f"Good subfolder found in {folder}. Using masked files from the good folder for processing.")
        crop_dir = f"{folder}/good"

    
    for fname in os.listdir(crop_dir):
        if fname.endswith("_masked.h5"):
            match = re.search(r'cluster_(\d+)_masked\.h5', fname)
            if match:
                cluster_id = match.group(1)
            else:
                cluster_id = re.sub(r"^\*cluster_", "", re.sub(r"_masked\.h5$", "", fname))
            if cluster_id in counts_map:
                data = counts_map[cluster_id]
                
                if not fname.startswith(prefix):
                    new_fname = f"{prefix}_{fname}"
                else:
                    new_fname = fname
                # Copy the file if combining and the new file doesn't already exist
                if combined:
                    shutil.copy(os.path.join(f"{folder}/masked", fname), os.path.join(combined_folder, new_fname))
                else:
                    # If not combining, just rename the file in place if it doesn't already have the prefix
                    if not fname.startswith(prefix):
                        os.rename(os.path.join(f"{folder}/masked", fname), os.path.join(f"{folder}/masked", new_fname))
                
                # Write to CSV
                with open(labels_combined_path, "a", encoding="utf-8") as f:
                    f.write(f"{new_fname},{data['pos']},{data['neg']},{data['size']}\n")
            
            else:
                print(f"Warning: Cluster ID {cluster_id} or filename {fname} not found in labels file for prefix {prefix}. Skipping counts for this file.")
    # now reoopen the csv and check for duplicate filenames and remove any, as appending means rerunning will create duplicates in the csv file, but we only want one entry per file. We can identify duplicates by looking for duplicate filenames in the first column, and we can keep the first occurrence and remove any subsequent occurrences.
    df_combined = pd.read_csv(labels_combined_path)
    df_combined = df_combined.drop_duplicates(subset=['cluster'], keep='first')
    df_combined.to_csv(labels_combined_path, index=False)

  
if __name__ == "__main__":
    if sys.argv[1:]:
        path = sys.argv[1]
    else:        
        path = '.'
    find_and_combine(path)