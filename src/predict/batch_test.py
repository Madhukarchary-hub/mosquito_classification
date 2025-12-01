import os
import torch
import timm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODEL_PATH = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\experiments\model1_baseline\model1_efficientnet_b2_best.pth"

TEST_FOLDER = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\test_mosquito_img_samples"

CLASS_NAMES = ["Ae_aegypti", "Ae_albopictus", "Ae_vexans", "Cx_quinquefasciatus"]

IMAGE_SIZE = 260

# ---------------------------------------------------------
# IMAGE TRANSFORMS
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
def load_model():
    print(f"\nLoading model from:\n{MODEL_PATH}")
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------------------------------------------------------
# PREDICT ONE IMAGE
# ---------------------------------------------------------
def predict_image(model, img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()

    return CLASS_NAMES[pred_class], float(probabilities[0][pred_class])

# ---------------------------------------------------------
# BATCH TEST
# ---------------------------------------------------------
def batch_test():
    model = load_model()
    results = []

    print("\nRunning batch test...\n")

    for true_class in os.listdir(TEST_FOLDER):
        class_path = os.path.join(TEST_FOLDER, true_class)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(class_path, file)

            pred_class, confidence = predict_image(model, img_path)

            results.append({
                "Image": file,
                "True Class": true_class,
                "Predicted Class": pred_class,
                "Confidence": round(confidence, 4)
            })

            print(f"{file}: True={true_class} → Pred={pred_class} ({confidence:.2f})")

    # Save results
    df = pd.DataFrame(results)
    output_csv = "batch_test_results.csv"
    df.to_csv(output_csv, index=False)

    print(f"\n📄 Results saved to: {output_csv}")

    # Show summary accuracy
    correct = df[df["True Class"] == df["Predicted Class"]].shape[0]
    total = df.shape[0]
    acc = correct / total * 100

    print(f"\nOverall Accuracy: {acc:.2f}%  ({correct}/{total} correct)\n")

    # Class-wise accuracy
    print("Class-wise accuracy:")
    for cls in CLASS_NAMES:
        sub = df[df["True Class"] == cls]
        if len(sub) > 0:
            cls_acc = (sub["True Class"] == sub["Predicted Class"]).mean() * 100
            print(f"  {cls}: {cls_acc:.2f}%  ({sum(sub['True Class'] == sub['Predicted Class'])}/{len(sub)})")

# ---------------------------------------------------------
if __name__ == "__main__":
    batch_test()
