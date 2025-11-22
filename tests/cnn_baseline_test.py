import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import sys
sys.path.insert(0, '.')
from src.datasets.tb_dataset import TBDataset

# Load CNN
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(512, 2)
cnn.load_state_dict(torch.load('results/checkpoints/cnn_best_v2.pt', map_location=torch.device('cpu')))
cnn.eval()

# Load test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = TBDataset(
    csv_file='splits/test_split.csv',
    root_dir='src/datasets/tb_dataset_crops',
    transform=transform
)

print(f"Testing CNN on {len(test_dataset)} images...\n")

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        output = cnn(img.unsqueeze(0))
        pred = output.argmax().item()
        
        correct += (pred == label)
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total} images...")

accuracy = 100 * correct / total
print(f"\n{'='*60}")
print(f"CNN-ONLY BASELINE")
print(f"{'='*60}")
print(f"Test Images:  {total}")
print(f"Correct:      {correct}")
print(f"Accuracy:     {accuracy:.2f}%")
print(f"{'='*60}")
