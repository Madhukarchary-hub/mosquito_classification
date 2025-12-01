import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

class MosquitoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.filepaths = df["filepath"].values
        self.labels = df["species"].astype("category").cat.codes.values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_model():
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)

    # Replace head for 4 species
    model.classifier[1] = nn.Linear(1408, 4)

    # UNFREEZE last 3 blocks + classifier (faster + strong accuracy)
    for name, param in model.named_parameters():
        if (
            name.startswith("features.6") or
            name.startswith("features.7") or
            name.startswith("features.8") or
            name.startswith("classifier")
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def train_model():

    ### ------------------------
    ### DATA AUGMENTATION
    ### ------------------------
    train_tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomResizedCrop(260, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = MosquitoDataset("data/csv/train.csv", train_tf)
    val_data = MosquitoDataset("data/csv/val.csv", val_tf)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    print("\nTrainable layers:")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    for t in trainable:
        print("  ", t)
    print(f"Total trainable params: {len(trainable)}\n")

    ### ------------------------
    ### OPTIMIZER + LOSS
    ### ------------------------
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    ### AMP (MIXED PRECISION)
    scaler = torch.cuda.amp.GradScaler()

    ### Early Stopping
    best_acc = 0
    patience = 5
    patience_counter = 0

    save_path = "experiments/model2_augmented/model2_efficientnet_b2_best.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ### ------------------------
    ### TRAIN LOOP
    ### ------------------------
    for epoch in range(1, 25):

        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            ### MIXED PRECISION TRAINING
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        ### ------------------------
        ### VALIDATION
        ### ------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=False):
                    preds = model(imgs)
                _, predicted = preds.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch} — Loss {running_loss:.2f}, Val Acc {val_acc:.4f}")

        ### ------------------------
        ### SAVE BEST MODEL
        ### ------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("🔥 Saved BEST model!")
        else:
            patience_counter += 1

        ### STOP IF NO IMPROVEMENT
        if patience_counter >= patience:
            print("\n⛔ Early stopping triggered (no improvement).\n")
            break

    print("\nTraining Completed!")


if __name__ == "__main__":
    train_model()
