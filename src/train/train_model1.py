import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# -----------------------------
# 1. Custom Dataset
# -----------------------------
class MosquitoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Encode species labels to numbers
        self.encoder = LabelEncoder()
        self.df['label'] = self.encoder.fit_transform(self.df['species'])
        self.labels = self.df['label'].values
        self.paths = self.df['filepath'].values
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# -----------------------------
# 2. Model (EfficientNet-B2)
# -----------------------------
def get_model(num_classes):
    model = timm.create_model("efficientnet_b2", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# -----------------------------
# 3. Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, device, epochs=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ----- Validation -----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "model1_efficientnet_b2_best.pth")
            best_val_acc = val_acc
            print("🔥 Saved new best model!")

    print("Training Completed.")

# -----------------------------
# 4. MAIN
# -----------------------------
if __name__ == "__main__":

    train_csv = "data/csv/train.csv"
    val_csv = "data/csv/val.csv"

    # Basic transforms
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    train_dataset = MosquitoDataset(train_csv, transform)
    val_dataset = MosquitoDataset(val_csv, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_classes = len(train_dataset.encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = get_model(num_classes).to(device)

    train_model(model, train_loader, val_loader, device, epochs=12)
