

import sys, os
import sys, os
import torch
import pandas as pd
from PIL import Image

from torchvision import transforms    

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

print("PYTHONPATH FIXED:", ROOT)




# ------------------------------------------------------------
# IMPORT MODELS (these folders exist in your project root)
# ------------------------------------------------------------
from src.models.model1_baseline_architecture import BaselineEffNetB2
from src.models.model2_augmented import EfficientNetB2 as Model2EffNetB2
from src.models.model3_multiview_architecture import MultiViewEffNetB2
from src.models.model4_distilled import DistilledMobileNetV2


# ------------------------------------------------------------
# IMAGE TRANSFORMS
# ------------------------------------------------------------
tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ------------------------------------------------------------
# LOAD MODEL HELPER
# ------------------------------------------------------------
def load_weights(model, path):
    print(f"▶ Loading weights: {path}")
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("   Missing keys:", len(missing))
    print("   Unexpected keys:", len(unexpected))
    return model.eval()


# ------------------------------------------------------------
# SINGLE IMAGE PREDICTION
# ------------------------------------------------------------
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = tf(img).unsqueeze(0)       # shape: 1×3×260×260
    logits = model(img)

    if isinstance(logits, tuple):    # Model-3 returns 3 outputs
        logits = logits[0]

    _, pred = torch.max(logits, dim=1)
    return int(pred.item())


# ------------------------------------------------------------
# MAIN BATCH PREDICTION LOGIC
# ------------------------------------------------------------
def run_batch_test():

    # Load your filtered batch CSV
    csv_path = "batch_test/batch_test_images_list.csv"
    df = pd.read_csv(csv_path)

    print("\n📌 Loaded dataset:", len(df), "images\n")

    # Load all 4 models
    print("📦 Loading models...\n")

    model1 = BaselineEffNetB2(num_classes=4)
    model1 = load_weights(model1, "experiments/model1_baseline/model1_efficientnet_b2_best.pth")

    model2 = Model2EffNetB2(num_classes=4)
    model2 = load_weights(model2, "experiments/model2_augmented/model2_efficientnet_b2_best.pth")

    model3 = MultiViewEffNetB2(num_classes=4)
    model3 = load_weights(model3, "experiments/model3_multiview/model3_efficientnet_b2_best.pth")

    model4 = DistilledMobileNetV2(num_classes=4)
    model4 = load_weights(model4, "experiments/model4_distilled/model4_student_best.pth")

    # Add empty prediction columns
    df["pred_model1"] = ""
    df["pred_model2"] = ""
    df["pred_model3"] = ""
    df["pred_model4"] = ""

    # Loop through all images
    print("\n🔍 Running predictions...\n")

    for idx, row in df.iterrows():
        img_path = row["filepath"]

        if idx % 500 == 0:
            print(f"   → Processed {idx} images...")

        df.at[idx, "pred_model1"] = predict_image(model1, img_path)
        df.at[idx, "pred_model2"] = predict_image(model2, img_path)

        # Model 3 uses 3 views → pass same image 3 times
        img = Image.open(img_path).convert("RGB")
        img = tf(img).unsqueeze(0)
        logits3 = model3(img, img, img)[0]
        df.at[idx, "pred_model3"] = int(logits3.argmax().item())

        df.at[idx, "pred_model4"] = predict_image(model4, img_path)

    # Save output
    out_path = "experiments/predictions/batch_test_results.csv"
    df.to_csv(out_path, index=False)
    print("\n🎉 DONE! Saved predictions to:", out_path)


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    run_batch_test()
