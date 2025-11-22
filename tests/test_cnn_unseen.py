"""Test CNN v3 on unseen images"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load CNN v3
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(512, 2)
cnn.load_state_dict(torch.load('results/checkpoints/cnn_best_v3.pt', map_location='cpu'))
cnn.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test unseen images
unseen_dir = 'src/datasets/unseen_images'
print("="*60)
print("CNN v3 - UNSEEN IMAGES TEST")
print("="*60)

correct = 0
total = 0

for img_file in sorted(os.listdir(unseen_dir)):
    if img_file.endswith(('.png', '.jpg')):
        img = Image.open(os.path.join(unseen_dir, img_file)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = cnn(tensor)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax().item()
            conf = probs.max().item()
        
        true_label = 1 if 'pos' in img_file else 0
        correct += (pred == true_label)
        total += 1
        
        status = '✅' if pred == true_label else '❌'
        print(f"{status} {img_file}: Pred={pred}, True={true_label}, Conf={conf:.2f}")

print("="*60)
print(f"CNN v3 Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
print("="*60)
