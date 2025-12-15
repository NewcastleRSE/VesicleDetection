import csv
import re
import os

CSV_PATH = "counts_clusters6_60_filtered.csv"        # path to your csv
FOLDER = "/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm_cluster_crops_allmasked/raw"        # folder with cluster_XX_raw.h5 files

# --- Read cluster numbers from CSV ---
csv_clusters = set()
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_clusters.add(int(row["cluster"]))

# --- Extract cluster numbers from filenames ---
filename_clusters = set()
pattern = re.compile(r"cluster_(\d+)_raw\.h5$")

for fname in os.listdir(FOLDER):
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


# --- export new CSV with only matching clusters ---
OUTPUT_CSV_PATH = "counts_clusters6_60_filtered_matched.csv"
with open(CSV_PATH, newline="") as infile, open(OUTPUT_CSV_PATH, "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if int(row["cluster"]) in both:
            # modify the cluster name to match filenames
            row["cluster"] = f"cluster_{row['cluster']}_masked.h5"
            writer.writerow(row)