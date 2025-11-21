import torch
from torchvision import transforms
from transformers import ViTForImageClassification
import sys
sys.path.insert(0, '.')
from src.datasets.tb_dataset import TBDataset

# Load ViT
vit = ViTForImageClassification.from_pretrained(
    'results/checkpoints/vit_best',
    num_labels=2,
    ignore_mismatched_sizes=True
)
vit.eval()

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

print(f"Testing ViT on {len(test_dataset)} images...\n")

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        outputs = vit(img.unsqueeze(0))
        pred = outputs.logits.argmax().item()
        
        correct += (pred == label)
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total} images...")

accuracy = 100 * correct / total
print(f"\n{'='*60}")
print(f"ViT-ONLY BASELINE")
print(f"{'='*60}")
print(f"Test Images:  {total}")
print(f"Correct:      {correct}")
print(f"Accuracy:     {accuracy:.2f}%")
print(f"{'='*60}")
