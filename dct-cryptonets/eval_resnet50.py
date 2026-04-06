import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==== CONFIG ====
VAL_DIR = "dataset/FaceForensic/val"
MODEL_PATH = "./checkpoint/faceforensic/resnet50_pth/best-0.980.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==== DATASET ====
dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== MODEL ====
model = models.resnet50()

# Adjust final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== EVALUATION ====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==== METRICS ====
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
