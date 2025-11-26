import os
from collections import Counter

ROOT = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\data\corpus_four_species"

ext_counts = Counter()
species_counts = Counter()

for species_folder in os.listdir(ROOT):
    species_dir = os.path.join(ROOT, species_folder)
    if not os.path.isdir(species_dir):
        continue

    for fname in os.listdir(species_dir):
        ext = os.path.splitext(fname)[1].lower()   # .png, .jpg, etc.
        ext_counts[ext] += 1
        species_counts[species_folder] += 1

print("Total items per species (all extensions):")
for k, v in species_counts.items():
    print(f"{k:20s}: {v}")

print("\nCounts by extension:")
for ext, cnt in ext_counts.items():
    print(f"{ext or '(no ext)':10s}: {cnt}")
