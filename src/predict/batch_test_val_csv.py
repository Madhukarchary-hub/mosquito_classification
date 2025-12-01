import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

# ---------------------------
# CONFIGURE THESE
# ---------------------------
CSV_PATH = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\data\csv\val.csv"
MODEL_PATH = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\experiments\model1_baseline\model1_efficientnet_b2_best.pth"
OUTPUT_CSV = "val_predictions.csv"

num_classes = 4
class_names = ["Ae-aegypti", "Ae-albopictus", "Ae-vexans", "Cx-quinquefasciatus"]

# ---------------------------
# MODEL LOADING
# ---------------------------
import timm
print("Loading model...")

model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------------------
# IMAGE TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------------------
# LOAD CSV
# ---------------------------
df = pd.read_csv(CSV_PATH)

print(f"\nLoaded {len(df)} rows from val.csv")

true_labels = df["species"].tolist()
image_paths = df["filepath"].tolist()

predicted_labels = []
predicted_probs = []
is_correct = []

# ---------------------------
# RUN BATCH PREDICTION
# ---------------------------
print("\nRunning batch evaluation...\n")

for img_path, true_label in tqdm(zip(image_paths, true_labels), total=len(df)):

    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print(f"Could not load {img_path}, skipping...")
        predicted_labels.append("ERROR")
        predicted_probs.append(0.0)
        is_correct.append(False)
        continue

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        pred_label = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

    predicted_labels.append(pred_label)
    predicted_probs.append(confidence)
    is_correct.append(pred_label == true_label)

# ---------------------------
# SAVE RESULTS
# ---------------------------
df_results = pd.DataFrame({
    "filepath": image_paths,
    "true_label": true_labels,
    "pred_label": predicted_labels,
    "confidence": predicted_probs,
    "correct": is_correct
})

df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

# ---------------------------
# ACCURACY CALCULATIONS
# ---------------------------
overall_acc = sum(is_correct) / len(is_correct)

print(f"\nOverall Accuracy: {overall_acc*100:.2f}% ({sum(is_correct)}/{len(is_correct)})\n")

# CLASS-WISE
print("Class-wise accuracy:")
for cls in class_names:
    cls_mask = [t == cls for t in true_labels]
    cls_total = sum(cls_mask)
    cls_correct = sum([is_correct[i] for i in range(len(is_correct)) if cls_mask[i]])
    acc = (cls_correct / cls_total) if cls_total > 0 else 0
    print(f"  {cls}: {acc*100:.2f}% ({cls_correct}/{cls_total})")
