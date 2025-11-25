import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# List all CNN model files to test
cnn_checkpoints = [
    'results/checkpoints/cnn_models/cnn_32_False_best.pt',
    'results/checkpoints/cnn_models/cnn_32_True_best.pt',
    'results/checkpoints/cnn_models/cnn_64_False_best.pt',
    'results/checkpoints/cnn_models/cnn_64_True_best.pt',
    'results/checkpoints/cnn_models/cnn_128_False_best.pt',
    'results/checkpoints/cnn_models/cnn_128_True_best.pt',
    'results/checkpoints/cnn_models/cnn_224_False_best.pt',
    'results/checkpoints/cnn_models/cnn_224_True_best.pt',
]

unseen_dir = 'src/datasets/unseen_images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

output_rows = []
header = ['model', 'image', 'pred', 'true', 'confidence', 'correct']

for model_path in cnn_checkpoints:
    model_name = os.path.basename(model_path)
    cnn = models.resnet18(weights=None)
    cnn.fc = nn.Linear(512, 2)
    cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
    cnn.eval()

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
                output = cnn(tensor)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax().item()
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

    # Add an empty separator row after each model
    output_rows.append(['']*len(header))

# Save all results to csv
with open('cnn_test_outputs.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_rows)

print("All model results saved to cnn_test_outputs.csv")
