

import os
import pandas as pd

ROOT = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\data\corpus_four_species"

rows = []

acceptable_ext = (".png", ".jpg", ".jpeg")

for species in os.listdir(ROOT):
    species_dir = os.path.join(ROOT, species)
    if not os.path.isdir(species_dir):
        continue

    print(f"Scanning: {species}")

    for fname in os.listdir(species_dir):
        if not fname.lower().endswith(acceptable_ext):
            continue

        filepath = os.path.join(species_dir, fname)

        # Parse file metadata
        parts = fname.split("_")
        try:
            phone = parts[0]
            sp = parts[1]
            p_id = parts[2]
            trial = parts[3]
            view = parts[4].split(".")[0]
        except:
            # Skip unexpected files
            continue

        rows.append([filepath, species, phone, sp, p_id, trial, view])

df = pd.DataFrame(rows, columns=[
    "filepath", "species", "phone", "sp_short", "specimen_id", "trial", "view"
])

os.makedirs("data/csv", exist_ok=True)
df.to_csv("data/csv/master.csv", index=False)
print("Master CSV created with:", len(df), "images")

# Train/val/test split
from sklearn.model_selection import train_test_split

unique_ids = df["specimen_id"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.35, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def subset(df, ids):
    return df[df["specimen_id"].isin(ids)]

subset(df, train_ids).to_csv("data/csv/train.csv", index=False)
subset(df, val_ids).to_csv("data/csv/val.csv", index=False)
subset(df, test_ids).to_csv("data/csv/test.csv", index=False)

print("train.csv, val.csv, test.csv created successfully!")
