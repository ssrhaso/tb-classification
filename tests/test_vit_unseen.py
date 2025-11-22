"""Test ViT v3 on unseen images"""
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import os

# Load ViT v3
vit = ViTForImageClassification.from_pretrained('results/checkpoints/vit_best_v3', num_labels=2)
vit.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test unseen images
unseen_dir = 'src/datasets/unseen_images'
print("="*60)
print("ViT v3 - UNSEEN IMAGES TEST")
print("="*60)

correct = 0
total = 0

for img_file in sorted(os.listdir(unseen_dir)):
    if img_file.endswith(('.png', '.jpg')):
        img = Image.open(os.path.join(unseen_dir, img_file)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = vit(tensor)
            logits = output.logits
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax().item()
            conf = probs.max().item()
        
        true_label = 1 if 'pos' in img_file else 0
        correct += (pred == true_label)
        total += 1
        
        status = '✅' if pred == true_label else '❌'
        print(f"{status} {img_file}: Pred={pred}, True={true_label}, Conf={conf:.2f}")

print("="*60)
print(f"ViT v3 Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
print("="*60)
