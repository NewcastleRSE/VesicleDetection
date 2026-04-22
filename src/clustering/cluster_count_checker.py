import csv
import re
import os


def cluster_count_checker(csv_path, folder):
    # --- Read cluster numbers from CSV ---
    csv_clusters = set()
    pattern = re.compile(r"cluster_(\d+)_(.*)\.h5$") 
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "cluster" not in reader.fieldnames:
            raise ValueError("Column 'cluster' not found in CSV")
        # get everything from the cluster column, and extract the number from it if it is in the form "cluster_XX_raw.h5"
        for row in reader:
            if row["cluster"].endswith(".h5"):  # are the clusters in the form "cluster_XX_raw.h5"?
                m = pattern.match(row["cluster"])
                print(f"\n pattern matches? {m}")
                if m:
                    csv_clusters.add(int(m.group(1)))
            else: # if they are raw ints, just add them directly
                csv_clusters.add(int(row["cluster"]))

    # --- Extract cluster numbers from filenames ---
    filename_clusters = set()

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            filename_clusters.add(int(m.group(1)))

    # --- Compare sets ---
    both = csv_clusters & filename_clusters
    only_csv = csv_clusters - filename_clusters
    only_files = filename_clusters - csv_clusters

    # --- Print results ---
    print("=== Summary ===")
    print(f"Clusters in both CSV and filenames ({len(both)}): {sorted(both)}")
    print()
    print(f"Clusters only in CSV ({len(only_csv)}): {sorted(only_csv)}")
    print()
    print(f"Clusters only in filenames ({len(only_files)}): {sorted(only_files)}")

if __name__ == "__main__":
    CSV_PATH = "/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_crops_maskedeps15ms10/cluster_labels.csv"        # path to your csv
    FOLDER = "/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_crops_maskedeps15ms10/raw"        # folder with cluster_XX_raw.h5 files

    cluster_count_checker(CSV_PATH, FOLDER)