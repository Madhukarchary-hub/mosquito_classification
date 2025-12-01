import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

# -------------------------------------------------------
#  FIX PYTHONPATH
# -------------------------------------------------------
ROOT = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification"
sys.path.append(ROOT)

from src.models.model1_baseline_architecture import BaselineEffNetB2
from src.models.model2_augmented import EfficientNetB2 as Model2EffNetB2
from src.models.model4_distilled import DistilledMobileNetV2


# -------------------------------------------------------
# IMAGE TRANSFORMS (same as training)
# -------------------------------------------------------
tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -------------------------------------------------------
# Load a model + weights
# -------------------------------------------------------
def load(model, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# -------------------------------------------------------
# Read test_lab_images folder
# -------------------------------------------------------
def load_lab_dataset(folder):
    records = []
    classes = sorted(os.listdir(folder))

    mapping = {cls: i for i, cls in enumerate(classes)}   # label → id

    for cls in classes:
        cls_path = os.path.join(folder, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.join(cls_path, fname),
                    "label": mapping[cls]
                })

    return pd.DataFrame(records), mapping


# -------------------------------------------------------
# Run predictions
# -------------------------------------------------------
def predict(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    return out.argmax(1).item()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    test_folder = os.path.join(ROOT, "batch_test", "test_lab_images")
    df, mapping = load_lab_dataset(test_folder)
    print(f"\n📌 Loaded {len(df)} lab-validation images")

    # Load only valid models
    print("\n📦 Loading model weights...")
    model1 = load(BaselineEffNetB2(num_classes=4),
                  os.path.join(ROOT, "experiments/model1_baseline/model1_efficientnet_b2_best.pth"))

    model2 = load(Model2EffNetB2(num_classes=4),
                  os.path.join(ROOT, "experiments/model2_augmented/model2_efficientnet_b2_best.pth"))

    model4 = load(DistilledMobileNetV2(num_classes=4),
                  os.path.join(ROOT, "experiments/model4_distilled/model4_student_best.pth"))

    # Store predictions
    df["pred_model1"] = 0
    df["pred_model2"] = 0
    df["pred_model4"] = 0

    print("\n🔍 Running predictions...")
    for i, row in df.iterrows():
        fp = row["filepath"]
        df.at[i, "pred_model1"] = predict(model1, fp)
        df.at[i, "pred_model2"] = predict(model2, fp)
        df.at[i, "pred_model4"] = predict(model4, fp)

        if i % 100 == 0:
            print(f" → Processed {i} images...")

    # Accuracy
    print("\n📊 Accuracy:")
    for col in ["pred_model1", "pred_model2", "pred_model4"]:
        acc = (df["label"] == df[col]).mean() * 100
        print(f"{col}: {acc:.2f}%")

    # Save results
    out_csv = os.path.join(ROOT, "experiments/predictions/lab_test_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved CSV → {out_csv}")


if __name__ == "__main__":
    main()
