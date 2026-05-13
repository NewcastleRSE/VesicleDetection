from os import path
import shutil
import sys
import pandas as pd
import os
import glob

def find_and_combine(path='.'):
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


def get_combined_counts_map(labels_file, ratings_path):
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
            
            pos_sum, neg_sum, size_sum = 0, 0, 0
            for cid in to_combine:
                cid = cid.strip()
                if cid in df_labels.index:
                    pos_sum += df_labels.loc[cid, 'positive_count']
                    neg_sum += df_labels.loc[cid, 'negative_count']
                    size_sum += df_labels.loc[cid, 'cluster_size']
            
            # Override the map with the summed values
            counts_map[target_id] = {'pos': pos_sum, 'neg': neg_sum, 'size': size_sum}
            
    return counts_map

def process_files(folder, combined_folder, counts_map, labels_combined_path, prefix, combined=True):
    for fname in os.listdir(f"{folder}/masked"):
        if fname.endswith("_masked.h5") and not fname.startswith(prefix):
            cluster_id = fname.replace("cluster_", "").replace("_masked.h5", "")
            
            if cluster_id in counts_map:
                data = counts_map[cluster_id]
                new_fname = f"{prefix}_{fname}"
                print(f"Processing file {fname} with cluster ID {cluster_id}. Counts: pos={data['pos']}, neg={data['neg']}, size={data['size']}")
                
                # Copy the file if combining and the new file doesn't already exist
                if combined:
                    shutil.copy(os.path.join(folder, fname), os.path.join(combined_folder, new_fname))
                
                # Write to CSV
                with open(labels_combined_path, "a", encoding="utf-8") as f:
                    f.write(f"{new_fname},{data['pos']},{data['neg']},{data['size']}\n")
            # labels_df['cluster'] = labels_df['cluster'].astype(str)  # Ensure cluster column is string for comparison
            # cluster_id = fname.replace("cluster_", "").replace("_masked.h5", "")
            # if cluster_id in labels_df['cluster'].values:
            #     row = labels_df[labels_df['cluster'] == cluster_id].iloc[0]
            #     with open(labels_combined_path, "a", encoding="utf-8") as f:
            #         f.write(f"{new_fname},{row['positive_count']},{row['negative_count']},{row['cluster_size']}\n")
            # elif fname in labels_df['cluster'].values:
            #     row = labels_df[labels_df['cluster'] == fname].iloc[0] 
            #     with open(labels_combined_path, "a") as f:
            #         f.write(f"{new_fname},{row['positive_count']},{row['negative_count']},{row['cluster_size']}\n")
            else:
                print(f"Warning: Cluster ID {cluster_id} or filename {fname} not found in labels file for prefix {prefix}. Skipping counts for this file.")


def process_and_combine_crops(prefix, base_path='.', combine=True):
    """This function will find folders with the prefix before convexhullmask or vesiclemask, it will rename the internal files to have the prefix appended if it isn't already,
    and then it will move the files into a combined folder to hold all the processed crops if they are not already there. It will also create a combined csv file with the counts for each cluster and the new filenames.
    The counts will be combined according to the "clusters to combine" column in the ratings file, and the new filenames will be in the format "{prefix}_cluster_{id}_masked.h5".

    Args:
        prefix (string): An identifier for the dataset that is used to find the relevant folders and files. For example, if prefix is "19-13_subvolume_0647-1670", it will look for folders like "19-13_subvolume_0647-1670_convexhullmask" and "19-13_subvolume_0647-1670_vesiclemask", 
                         and it will look for files like "cluster_4_masked.h5" inside those folders. It will then create new files like "19-13_subvolume_0647-1670_cluster_4_masked.h5" in a combined folder, and it will create a csv file with the counts for each cluster and the new filenames.
    """
    # find the relevant folders and file
    convex_folder = glob.glob(f"{base_path}/{prefix}*convexhull*")
    if not convex_folder:
        print(f"No convex hull folder found for prefix {prefix} in {base_path}.")
        convex_folder = None
    else:        
        convex_folder = convex_folder[0]
    vesicle_folder = glob.glob(f"{base_path}/{prefix}*vesicle*")
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
   
    if os.path.exists(f"{base_path}/{prefix}ratings.csv"):
        print(f"Ratings file found for prefix {prefix}. Combining counts according to ratings.")
        counts_map = get_combined_counts_map(labels_file, f"{base_path}/{prefix}ratings.csv")
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

  
if __name__ == "__main__":
    if sys.argv[1:]:
        path = sys.argv[1]
    else:        
        path = '.'
    find_and_combine(path)