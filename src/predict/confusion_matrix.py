import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Load validation predictions
# -----------------------------
csv_path = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\experiments\predictions\val_predictions.csv"# Adjust if needed

df = pd.read_csv(csv_path)

# Expected columns:
# filepath, true_label, pred_label, pred_prob
if not {"true_label", "pred_label"}.issubset(df.columns):
    raise ValueError("CSV must contain true_label and pred_label columns.")

y_true = df["true_label"]
y_pred = df["pred_label"]

# -----------------------------
# Compute confusion matrix
# -----------------------------
labels = sorted(df["true_label"].unique())

cm = confusion_matrix(y_true, y_pred, labels=labels)

# -----------------------------
# Plot confusion matrix
# -----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Mosquito Species Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# Print Classification Report
# -----------------------------
print("\n===== Classification Report =====\n")
print(classification_report(y_true, y_pred, labels=labels))
