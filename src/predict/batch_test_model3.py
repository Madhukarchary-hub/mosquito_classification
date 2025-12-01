import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------------------------------------
# MULTI-VIEW VALIDATION DATASET
# ----------------------------------------------------------
class MultiViewValDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.filepaths = df["filepath"].values
        self.labels = df["species"].astype("category").cat.codes.values
        self.tf = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")

        # 3 identical views (no augmentation)
        v1 = self.tf(img)
        v2 = self.tf(img)
        v3 = self.tf(img)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return v1, v2, v3, label


# ----------------------------------------------------------
# MODEL-3 MULTIVIEW ARCHITECTURE
# ----------------------------------------------------------
class MultiViewEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        base = efficientnet_b2(weights=weights)
        base.classifier[1] = nn.Identity()
        self.backbone = base

        # merge 3 × 1408 = 4224
        self.fc = nn.Linear(1408 * 3, 4)

    def forward(self, v1, v2, v3):
        f1 = self.backbone(v1)
        f2 = self.backbone(v2)
        f3 = self.backbone(v3)
        merged = torch.cat([f1, f2, f3], dim=1)
        return self.fc(merged)


# ----------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ----------------------------------------------------------
def run_batch_test():

    print("\n🔍 Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    # ---------------------------
    # LOAD MODEL-3
    # ---------------------------
    model = MultiViewEffNet().to(device)
    model.load_state_dict(torch.load(
        "experiments/model3_multiview/model3_efficientnet_b2_best.pth",
        map_location=device
    ))
    model.eval()

    # ---------------------------
    # TRANSFORM
    # ---------------------------
    tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---------------------------
    # LOAD VAL DATA
    # ---------------------------
    val_data = MultiViewValDataset("data/csv/val.csv", tf)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    all_preds = []
    all_labels = []

    # ---------------------------
    # RUN PREDICTIONS
    # ---------------------------
    with torch.no_grad():
        for v1, v2, v3, labels in val_loader:
            v1, v2, v3 = v1.to(device), v2.to(device), v3.to(device)
            labels = labels.to(device)

            outputs = model(v1, v2, v3)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------------------------------------------------
    # STATS
    # ------------------------------------------------------
    print("\n📊 Classification Report:\n")
    target_names = ["Ae-aegypti", "Ae-albopictus", "Ae-vexans", "Cx-quinquefasciatus"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    overall_acc = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"\n🔵 Validation Accuracy: {overall_acc * 100:.2f}%\n")

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)

    # Save CSV
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        "model3_val_confusion_matrix.csv"
    )
    print("📁 Saved CSV → model3_val_confusion_matrix.csv")

    # Save Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names,
                cmap="Blues")
    plt.title("Model-3 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("model3_val_confusion_matrix.png")
    plt.close()

    print("🖼 Saved heatmap → model3_val_confusion_matrix.png\n")

    print("✅ Batch testing completed!")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    run_batch_test()
