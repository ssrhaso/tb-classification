import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import os
import csv

# List all ViT model folders to test
vit_model_dirs = [
    'results/checkpoints/vit_models/vit_224_False_best',
    'results/checkpoints/vit_models/vit_224_True_best',
]

unseen_dir = 'src/datasets/unseen_images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

output_rows = []
header = ['model', 'image', 'pred', 'true', 'confidence', 'correct']

for model_dir in vit_model_dirs:
    model_name = os.path.basename(model_dir)
    vit = ViTForImageClassification.from_pretrained(model_dir, num_labels=2)
    vit.eval()

    correct = 0
    total = 0

    print("="*60)
    print(f"{model_name} - UNSEEN IMAGES TEST")
    print("="*60)

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
            is_correct = int(pred == true_label)
            correct += is_correct
            total += 1

            status = '✅' if is_correct else '❌'
            print(f"{status} {img_file}: Pred={pred}, True={true_label}, Conf={conf:.2f}")

            output_rows.append([
                model_name, img_file, pred, true_label, f"{conf:.2f}", is_correct
            ])

    print("="*60)
    print(f"{model_name} Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    print("="*60)

    # Add empty separator row after each model
    output_rows.append(['']*len(header))

# Save all results to csv
with open('vit_test_outputs.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_rows)

print("All model results saved to vit_test_outputs.csv")
