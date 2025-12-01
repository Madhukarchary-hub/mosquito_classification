import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

class MosquitoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.filepaths = df["filepath"].values
        self.labels = df["species"].astype("category")  # Keep category
        self.label_codes = self.labels.cat.codes        # Convert to int
        self.classes = list(self.labels.cat.categories) # Class names
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        label = self.label_codes[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(checkpoint_path):
    print("Loading EfficientNet-B2 model...")

    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)

    # Replace classifier
    model.classifier[1] = nn.Linear(1408, 4)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model.eval()
    return model


def batch_validate():

    checkpoint = "experiments/model2_augmented/model2_efficientnet_b2_best.pth"

    model = load_model(checkpoint)

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    dataset = MosquitoDataset("data/csv/val.csv", transform)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert index to class names
    class_names = dataset.classes

    # PRINT ACCURACY
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🔍 Validation Accuracy: {accuracy * 100:.2f}%\n")

    # PRINT CLASSIFICATION REPORT
    print("\n📌 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # SAVE CONFUSION MATRIX CSV
    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("model2_val_confusion_matrix.csv")

    print("\n📁 Confusion matrix saved → model2_val_confusion_matrix.csv")


if __name__ == "__main__":
    batch_validate()
