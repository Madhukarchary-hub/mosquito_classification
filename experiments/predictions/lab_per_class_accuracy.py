import pandas as pd
import os

ROOT = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification"
csv_path = os.path.join(ROOT, "experiments/predictions/lab_test_results.csv")

df = pd.read_csv(csv_path)

# -------------------------------------------------------
# MODEL columns (edit if needed)
# -------------------------------------------------------
model_cols = ["pred_model1", "pred_model2", "pred_model4"]

# -------------------------------------------------------
# Create mapping reverse dict
# label → species
# -------------------------------------------------------
# Example:
# Aedes_aegypti → 0
# Aedes_albopictus → 1
# Culex_quinquefasciatus → 2

# Extract mapping from folder structure saved earlier if needed
# But for now detect labels automatically:

label_to_species = {}

for fp, lab in zip(df["filepath"], df["label"]):
    species = fp.split("\\")[-2]  # folder name
    label_to_species[lab] = species

print("\nDetected species mapping:")
for k, v in label_to_species.items():
    print(f"  Label {k} → {v}")

# -------------------------------------------------------
# Calculate per-class accuracy
# -------------------------------------------------------
print("\n📊 Per-Class Accuracy:\n")

for model in model_cols:
    print(f"\n=== {model} ===")
    for class_id, species in label_to_species.items():
        sub = df[df["label"] == class_id]
        if len(sub) == 0:
            continue
        acc = (sub["label"] == sub[model]).mean() * 100
        print(f"{species:25s}: {acc:5.2f}%")
