#%%
import os
import csv
import random

#%%
masks_dir = "data/masks"
xrays_dir = "data/xrays"

whole_dataset_csv = "dataset.csv"

xray_files = sorted(
    os.listdir(xrays_dir),
    key=lambda x: int(os.path.splitext(x)[0])
)

#%%
with open(whole_dataset_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['xray_path', 'mask_path'])

    for xray_name in xray_files:
        xray_path = os.path.join(xrays_dir, xray_name)
        mask_path = os.path.join(masks_dir, xray_name)

        if os.path.exists(mask_path):
            writer.writerow([xray_path, mask_path])
        else:
            writer.writerow([xray_path, None])

#%%
input_csv = "data/CSVs/dataset.csv"
output_dir = "data/CSVs"
os.makedirs(output_dir, exist_ok=True)

# Read all rows
with open(input_csv, newline='') as f:
    rows = list(csv.DictReader(f))

# Split into labeled and unlabeled
labeled = [r for r in rows if r['mask_path'] not in ('None', '', None)]
unlabeled = [r for r in rows if r['mask_path'] in ('None', '', None)]

# Shuffle labeled data
random.seed(42)  # fixed for reproducibility
random.shuffle(labeled)

# 80/20 split
split = int(0.8 * len(labeled))
train, val = labeled[:split], labeled[split:]


def write_csv(path, data, fields):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            filtered = {k: row[k] for k in fields}  # only keep wanted keys
            writer.writerow(filtered)

# Save all three splits
write_csv(os.path.join(output_dir, 'train.csv'), train, ['xray_path', 'mask_path'])
write_csv(os.path.join(output_dir, 'val.csv'), val, ['xray_path', 'mask_path'])
write_csv(os.path.join(output_dir, 'test.csv'), unlabeled, ['xray_path'])

print("✅ CSVs saved in data/CSVs/: train.csv, val.csv, test.csv")
