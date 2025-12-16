import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "best_currency_efficientnet.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 9

CLASS_NAMES = ['1', '10', '10 (new)', '100', '20', '20 (new)', '200', '5', '50']

MEAN = [0.5270947813987732, 0.49859872460365295, 0.4817364811897278]
STD  = [0.25115787982940674, 0.2486504316329956, 0.2477809339761734]


def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict(image_path):
    model = load_model()
    image = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    print("Predicted Currency:", CLASS_NAMES[pred.item()])
    print(f"Confidence: {conf.item():.2%}")


if __name__ == "__main__":
    image_path = r"C:\Users\User\Music\download (4).jpeg"
    predict(image_path)
