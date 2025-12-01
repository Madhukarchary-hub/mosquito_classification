import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import os

# ---------------------------------------------------------
# CONFIG — your model path is set here
# ---------------------------------------------------------
DEFAULT_MODEL_PATH = r"C:\Users\mkuchavaram\OneDrive - Texas A&M University-Corpus Christi\Documents\mosquito_classification\experiments\model1_baseline\model1_efficientnet_b2_best.pth"
IMAGE_SIZE = 260

CLASS_NAMES = ["Ae-aegypti", "Ae-albopictus", "Ae-vexans", "Cx-quinquefasciatus"]

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
def load_model(weights_path):
    print(f"\nLoading model weights from:\n{weights_path}")

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
def predict(image_path, weights_path=DEFAULT_MODEL_PATH):

    if not os.path.exists(image_path):
        print("\n❌ ERROR: Image file not found.")
        return

    if not os.path.exists(weights_path):
        print("\n❌ ERROR: Model file not found:")
        print(weights_path)
        return

    model = load_model(weights_path)

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print("\n==============================")
    print(f" 🦟 Predicted Class: {CLASS_NAMES[predicted_class]}")
    print("==============================\n")

# ---------------------------------------------------------
# USER INPUT MODE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n=== MOSQUITO CLASSIFIER ===")
    print("Paste the FULL PATH of the mosquito image you want to classify.\n")

    image_path = input("📁 Enter image path: ").strip().strip('"')

    if image_path == "":
        print("\n❌ No image path provided. Exiting.\n")
        exit()

    predict(image_path)
