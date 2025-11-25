import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import os
import sys
import csv

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from src.adaptive_system import AdaptiveClassifier, CONFIG

TEST_CONFIG = {
    'unseen_images_dir': 'src/datasets/unseen_images',
    'threshold': 0.9,
}

if __name__ == '__main__':
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['resolution'], CONFIG['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("LOADING ADAPTIVE CLASSIFIER")
    adaptive = AdaptiveClassifier(
        cnn_checkpoint_path = CONFIG['cnn_checkpoint'],
        vit_checkpoint_path = CONFIG['vit_checkpoint'],
        cnn_architecture = CONFIG['cnn_architecture'],
        vit_architecture = CONFIG['vit_architecture'],
        threshold = TEST_CONFIG['threshold'],
        resolution = CONFIG['resolution'],
        device = CONFIG['device'],
        num_classes = CONFIG['num_classes'],
        transform = test_transform
    )

    image_files = sorted([
        f for f in os.listdir(TEST_CONFIG['unseen_images_dir'])
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"FOUND {len(image_files)} UNSEEN IMAGES TO TEST")
    
    cnn_correct = 0
    vit_correct = 0
    adaptive_correct = 0

    print("-" * 40)
    print("CLASSIFYING UNSEEN IMAGES...")
    print("-" * 40)

    output_rows = []
    header = [
        'image', 'true_label', 'cnn_pred', 'cnn_conf', 'vit_pred', 'vit_conf',
        'adaptive_pred', 'adaptive_conf', 'routed_to', 'latency_ms', 'cnn_correct', 'vit_correct', 'adaptive_correct'
    ]

    for img_file in image_files:
        img_path = os.path.join(TEST_CONFIG['unseen_images_dir'], img_file )
        image = Image.open(img_path).convert('RGB')
        tensor = test_transform(image) 

        adaptive_prediction, routed_to, adaptive_confidence, latency, stats = adaptive.predict(tensor)
        cnn_prediction = stats['cnn_prediction']
        cnn_confidence = stats['cnn_confidence']    
        vit_prediction, vit_confidence, _ = adaptive._predict_vit(tensor)

        true_label = 1 if 'pos' in img_file.lower() else 0

        cnn_corr = int(cnn_prediction == true_label)
        vit_corr = int(vit_prediction == true_label)
        adaptive_corr = int(adaptive_prediction == true_label)

        cnn_correct += cnn_corr
        vit_correct += vit_corr
        adaptive_correct += adaptive_corr

        output_rows.append([
            img_file, true_label, cnn_prediction, f"{cnn_confidence:.2f}", 
            vit_prediction, f"{vit_confidence:.2f}", adaptive_prediction, f"{adaptive_confidence:.2f}", 
            routed_to, f"{latency:.3f}", cnn_corr, vit_corr, adaptive_corr
        ])

        print("-" * 80)
        print(
            f"IMAGE: {img_file} | TRUE LABEL: {true_label} | CNN PRED: {cnn_prediction} (Conf: {cnn_confidence:.2f}) | "
            f"VIT PRED: {vit_prediction} (Conf: {vit_confidence:.2f}) | ADAPTIVE PRED: {adaptive_prediction} (Conf: {adaptive_confidence:.2f}) | "
            f"ROUTED TO: {routed_to} | LATENCY: {latency:.3f}ms"
        )
        print("-" * 80)
    
    with open('adaptive_test_outputs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)
    
    total_images = len(image_files)
    print("\n" + "-" * 60)
    print("TESTING COMPLETE (AUGMENTATION)")
    print("-" * 60)
    print(f"CNN ACCURACY: {cnn_correct}/{total_images} = {100 * cnn_correct / total_images:.2f}%")
    print(f"VIT ACCURACY: {vit_correct}/{total_images} = {100 * vit_correct / total_images:.2f}%")
    print(f"ADAPTIVE CLASSIFIER ACCURACY: {adaptive_correct}/{total_images} = {100 * adaptive_correct / total_images:.2f}%")
    print("-" * 60)
    print("All adaptive results saved to adaptive_test_outputs.csv")
