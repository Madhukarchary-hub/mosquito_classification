import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import time


# -----------------------------------------------------
# MULTI-VIEW DATASET (3 augmentations per sample)
# -----------------------------------------------------
class MultiViewDataset(Dataset):
    def __init__(self, csv_path, transform_main, transform_view1, transform_view2):
        df = pd.read_csv(csv_path)
        self.filepaths = df["filepath"].values
        self.labels = df["species"].astype("category").cat.codes.values

        self.tf_main = transform_main
        self.tf_view1 = transform_view1
        self.tf_view2 = transform_view2

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")

        # 3 views (augmentations)
        v1 = self.tf_main(img)
        v2 = self.tf_view1(img)
        v3 = self.tf_view2(img)

        # IMPORTANT FIX: convert label → LongTensor
        label = int(self.labels[idx])
        label = torch.tensor(label, dtype=torch.long)

        return (v1, v2, v3, label)


# -----------------------------------------------------
# MODEL DEFINITION (EFFNET-B2 WITH 3-VIEW FUSION)
# -----------------------------------------------------
class MultiViewEffNet(nn.Module):
    def __init__(self):
        super().__init__()

        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        base = efficientnet_b2(weights=weights)

        # Replace classifier for 4 classes
        base.classifier[1] = nn.Identity()
        self.backbone = base

        # 3 × 1408 features = 4224
        self.fc = nn.Linear(1408 * 3, 4)

    def forward(self, v1, v2, v3):
        f1 = self.backbone(v1)
        f2 = self.backbone(v2)
        f3 = self.backbone(v3)

        merged = torch.cat([f1, f2, f3], dim=1)
        out = self.fc(merged)
        return out


# -----------------------------------------------------
# TRAINING FUNCTION
# -----------------------------------------------------
def train_model():

    # ------------------------------
    # DATA AUGMENTATION
    # ------------------------------
    tf_main = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tf_view1 = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    tf_view2 = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomRotation(20),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = MultiViewDataset(
        "data/csv/train.csv",
        tf_main, tf_view1, tf_view2
    )

    val_dataset = MultiViewDataset(
        "data/csv/val.csv",
        tf_main, tf_main, tf_main
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ------------------------------
    # DEVICE
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 DEVICE: {device}\n")

    model = MultiViewEffNet().to(device)

    # ------------------------------
    # TRAINABLE LAYERS CHECK
    # ------------------------------
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    print("Trainable layers:")
    for t in trainable:
        print("  ", t)
    print(f"\nTotal trainable params: {len(trainable)}\n")

    # ------------------------------
    # OPTIMIZER + LOSS
    # ------------------------------
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    for epoch in range(1, 16):

        start = time.time()

        model.train()
        running_loss = 0

        for v1, v2, v3, labels in train_loader:
            v1, v2, v3, labels = v1.to(device), v2.to(device), v3.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(v1, v2, v3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ------------------------------
        # VALIDATION
        # ------------------------------
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for v1, v2, v3, labels in val_loader:
                v1, v2, v3, labels = v1.to(device), v2.to(device), v3.to(device), labels.to(device)
                preds = model(v1, v2, v3)
                _, predicted = preds.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        epoch_time = time.time() - start

        print(f"Epoch {epoch}: Loss={running_loss:.2f}, Val Acc={val_acc:.4f}, Time={epoch_time:.1f}s")

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "experiments/model3_multiview/model3_efficientnet_b2_best.pth")
            print("🔥 Saved BEST model!\n")


    print("\n🎉 Training Complete!\n")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    train_model()
