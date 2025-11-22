import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


from src.adaptive_system import AdaptiveClassifier, CONFIG

TEST_CONFIG = {
    'unseen_images_dir': 'src/datasets/unseen_images',
    'threshold': 0.9,
}



# ENTRY
if __name__ == '__main__':
    
    # SET UP TRANSFORM
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['resolution'], CONFIG['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # ADAPTIVE CLASSIFIER (LOADS PRETRAINED VIT MODEL INSIDE)
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
    
    # LIST OF TEST IMAGES
    image_files = sorted([
        f for f in os.listdir(TEST_CONFIG['unseen_images_dir'])
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"FOUND {len(image_files)} UNSEEN IMAGES TO TEST")
    
    # TRACK RESULTS
    cnn_correct = 0
    vit_correct = 0
    adaptive_correct = 0
    
    print("-" * 40)
    print("CLASSIFYING UNSEEN IMAGES...")
    print("-" * 40)
    
    for img_file in image_files:
        img_path = os.path.join(TEST_CONFIG['unseen_images_dir'], img_file  )
        
        # LOAD
        image = Image.open(img_path).convert('RGB')
        tensor = test_transform(image) 
        
        # PREDICTIONS FROM ADAPTIVE CLASSIFIER
        adaptive_prediction, routed_to, adaptive_confidence, latency, stats = adaptive.predict(tensor)
        
        # CNN ONLY
        cnn_prediction = stats['cnn_prediction']
        cnn_confidence = stats['cnn_confidence']    
        
        # VIT ONLY
        vit_prediction, vit_confidence, _ = adaptive._predict_vit(tensor)
        
        # DETERMINE TRUE LABEL (BASED ON FILENAME)
        true_label = 1 if 'pos' in img_file.lower() else 0
        label_name = 'Positive' if true_label == 1 else 'Negative'
        
        
        # ACCURACY
        cnn_correct += (cnn_prediction == true_label)
        vit_correct += (vit_prediction == true_label)
        adaptive_correct += (adaptive_prediction == true_label)
        
        # PRINT RESULTS
        print("-" * 80)
        print(f"IMAGE: {img_file} | TRUE LABEL: {label_name} | "
              f"TRUE LABEL: {true_label} | "
              f"CNN PRED: {cnn_prediction} (Conf: {cnn_confidence:.2f}) | "
              f"VIT PRED: {vit_prediction} (Conf: {vit_confidence:.2f}) | "
              f"ADAPTIVE PRED: {adaptive_prediction} (Conf: {adaptive_confidence:.2f}) | "
              f"ROUTED TO: {routed_to} | LATENCY: {latency:.3f}ms")
        print("-" * 80)
        
        
    # SUMMARY
    total_images = len(image_files)
    print("\n" + "-" * 60)
    print("TESTING COMPLETE")
    print("-" * 60)
    print(f"CNN ACCURACY: {cnn_correct}/{total_images} = "
          f"{100 * cnn_correct / total_images:.2f}%")
    print(f"VIT ACCURACY: {vit_correct}/{total_images} = "
          f"{100 * vit_correct / total_images:.2f}%")
    print(f"ADAPTIVE CLASSIFIER ACCURACY: {adaptive_correct}/{total_images} = "
          f"{100 * adaptive_correct / total_images:.2f}%")
    print("-" * 60)
    
    
    
    
        
        