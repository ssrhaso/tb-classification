""" TRAIN CNN ResNet on TB Dataset """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import argparse
import time
import os

# root (tb-classification), assuming train_cnn.py is in src/train/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "datasets", "tb_dataset_crops")
# Import your CNN model
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.tb_dataset import TBDataset, create_data_splits


if __name__ == '__main__':  # PREVENT THREADING ISSUES ON WINDOWS
    # CONFIGURATION
    EXPERIMENT = 1 # 1, 2, 3 ,4

    CONFIGS = {
        1: {'model' : 'resnet18', 'resolution': 224, 'pretrained': True, 'augment': False},
        2: {'model' : 'resnet50', 'resolution': 224, 'pretrained': False, 'augment': False},
        3: {'model' : 'resnet50', 'resolution': 128, 'pretrained': True, 'augment': True},
        4: {'model' : 'resnet18', 'resolution': 128, 'pretrained': False, 'augment': True},
    }
    config = CONFIGS[EXPERIMENT]

    print("-"*60)
    print("TRAINING CNN RESNET ON TB DATASET")
    print("-"*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 1. Prepare TB dataset
    print("\n1. Preparing TB dataset...")

    # TRANSFORMS
    if config['augment']:   # FOR AUGMENTATION
        train_transform = transforms.Compose([
        transforms.Resize((config['resolution'], config['resolution'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config['resolution'], config['resolution'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    ])
        
    val_transform = transforms.Compose([
        transforms.Resize((config['resolution'], config['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # CREATE SPLITS IF NOT EXIST
    if not os.path.exists('splits/train_split.csv'):
        create_data_splits(data_dir= DATA_DIR, seed=42)
        
    # DATASETS
    trainset = TBDataset(csv_file = 'splits/train_split.csv', root_dir = DATA_DIR, transform = train_transform)
    valset = TBDataset(csv_file = 'splits/val_split.csv', root_dir = DATA_DIR, transform = val_transform)
    print(f"TRAIN : {len(trainset)}, VAL : {len(valset)}")

    # DATALOADERS
    trainloader = DataLoader(trainset, batch_size = 32, shuffle = True, num_workers = 4)
    valloader = DataLoader(valset, batch_size = 128, shuffle = False, num_workers = 4)


    print(f" TRAINING set: {len(trainset)} images")
    print(f" VALIDATION set: {len(valset)} images")

    # 2. SETUP TRAINING
    print("\n2. SETTING UP TRAINING...")

    if config['model'] == 'resnet18':
        if config['pretrained']:
            model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights = None)
        model.fc = nn.Linear(in_features = 512, out_features = 2)   

    elif config['model'] == 'resnet50':
        if config['pretrained']:
            model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights = None)
        model.fc = nn.Linear(in_features = 2048, out_features = 2)
    model = model.to(device)
    print(f" MODEL LOADED : {config['model']} | PRETRAINED : {config['pretrained']}")



    # LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAINING LOOP
    EPOCHS = 1
    print(f" EPOCHS : {EPOCHS}")
    print(f" STARTING TRAINING... \n")


    # Create checkpoint directory
    os.makedirs("results/checkpoints", exist_ok=True)

    # 3. Training loop
    print("\n3. Starting training...")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print("-"*60)
        
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        

        train_loss = train_loss / len(trainloader)
        train_accuracy = 100 * train_correct / train_total
        
        
        # EVALUATE ON VALIDATION SET
        print("EVALUATING :")
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(valloader)
        val_accuracy = 100 * val_correct / val_total
        
        
        # METRICS:
        print(f" Epoch [{epoch+1}/{EPOCHS}] ")
        print(f"TRAIN LOSS : {train_loss:.4f} | TRAIN ACC : {train_accuracy:.2f}%")     
        print(f"VAL LOSS : {val_loss:.4f} | VAL ACC : {val_accuracy:.2f}%")
        
        
        # SAVE BEST MODEL
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "results/checkpoints/cnn_best.pt")
            print(f"BEST MODEL SAVED (ACCURACY : {best_val_acc:.2f}%)")
        print()


    print("\n" + "-"*60)
    print("TRAINING COMPLETE!")
    print("-"*60)
