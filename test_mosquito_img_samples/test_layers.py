from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

for name, _ in model.named_parameters():
    print(name)
