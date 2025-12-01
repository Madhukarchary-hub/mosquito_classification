import pandas as pd

# Load prediction file
df = pd.read_csv("experiments/predictions/batch_test_results.csv")

# Map class labels to numeric IDs
label_map = {
    "aegypti": 0,
    "albopictus": 1,
    "anopheles": 2,
    "culex": 3,
}

df["true_id"] = df["class_label"].map(label_map)

# Accuracy function
def accuracy(col):
    return (df[col] == df["true_id"]).mean() * 100

acc1 = accuracy("pred_model1")
acc2 = accuracy("pred_model2")
acc3 = accuracy("pred_model3")
acc4 = accuracy("pred_model4")

print(f"Model 1 Accuracy: {acc1:.2f}%")
print(f"Model 2 Accuracy: {acc2:.2f}%")
print(f"Model 3 Accuracy: {acc3:.2f}%")
print(f"Model 4 Accuracy: {acc4:.2f}%")

summary = pd.DataFrame({
    "model": ["model1","model2","model3","model4"],
    "accuracy": [acc1, acc2, acc3, acc4]
})

summary.to_csv("experiments/predictions/batch_accuracy_summary.csv", index=False)
print("\n✓ Accuracy summary saved!")
