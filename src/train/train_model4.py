import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# ---------------------------------------
# DATASET
# ---------------------------------------
class MosquitoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.filepaths = df["filepath"].values
        self.labels = df["species"].astype("category").cat.codes
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        label = int(self.labels[idx])  # convert to integer

        if self.transform:
            img = self.transform(img)

        return img, label


# ---------------------------------------
# LOAD TEACHER MODEL (MODEL-3)
# ---------------------------------------
from models.model3_multiview_architecture import MultiViewEffNetB2

def load_teacher_model(weights_path):
    model = MultiViewEffNetB2(num_classes=4)

    # Relaxed loading
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("⚠ Missing keys:", missing)
    print("⚠ Unexpected keys:", unexpected)
    print("✔ Loaded teacher model with relaxed matching")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


# ---------------------------------------
# STUDENT MODEL (MobileNet-V2)
# ---------------------------------------
def build_student_model():
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    student = mobilenet_v2(weights=weights)

    # Replace classifier → 4 classes
    student.classifier[1] = nn.Linear(1280, 4)

    return student


# ---------------------------------------
# KNOWLEDGE DISTILLATION LOSS
# ---------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.7):
    """
    student_logits: student output
    teacher_logits: teacher output
    labels: true labels
    T: temperature
    alpha: balance between soft loss and hard loss
    """

    # Softened teacher predictions
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    student_soft = F.log_softmax(student_logits / T, dim=1)

    # KL divergence
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

    # Cross entropy with real labels
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ---------------------------------------
# TRAINING LOOP
# ---------------------------------------
def train_model4():
    print("\n🔥 Training MODEL-4 (Distilled MobileNetV2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📌 Device:", device)

    # Ensure output folder exists
    os.makedirs("experiments/model4_distilled", exist_ok=True)

    # Transforms
    tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_data = MosquitoDataset("data/csv/train.csv", tf)
    val_data   = MosquitoDataset("data/csv/val.csv", tf)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)

    # Load Teacher Model-3
    teacher = load_teacher_model(
        "experiments/model3_multiview/model3_efficientnet_b2_best.pth"
    ).to(device)

    # Student
    student = build_student_model().to(device)
    optimizer = Adam(student.parameters(), lr=3e-4)

    best_acc = 0

    # ------------------------------
    # EPOCH LOOP
    # ------------------------------
    for epoch in range(1, 16):
        student.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()      # FIX: must be long

            optimizer.zero_grad()

            # Teacher output
            with torch.no_grad():
                teacher_logits = teacher(imgs, imgs, imgs)  # simple 3-view version

            # Student output
            student_logits = student(imgs)

            # Distillation loss
            loss = distillation_loss(student_logits, teacher_logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ------------------------------
        # VALIDATION
        # ------------------------------
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = student(imgs)
                _, predicted = preds.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Val Acc={acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(),
                       "experiments/model4_distilled/model4_student_best.pth")
            print("🔥 Saved distilled model!")

    print("\n🎉 Training complete!")
    print("🏆 Best student accuracy:", best_acc)



if __name__ == "__main__":
    train_model4()
