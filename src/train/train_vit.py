"""TRAIN VIT on TB Dataset"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'src', 'datasets', 'tb_dataset_crops')
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.datasets.tb_dataset import TBDataset, create_data_splits


if __name__ == '__main__':  # PREVENT THREADING ISSUES ON WINDOWS

	EXPERIMENT = 1

	CONFIGS = {
		1: {'model': 'vit-base', 'resolution': 224, 'pretrained': True, 'augment': True},
	}
	config = CONFIGS[EXPERIMENT]

	print('-' * 60)
	print('TRAINING VIT ON TB DATASET')
	print('-' * 60)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')

	# 1. LOAD PRETRAINED VIT MODEL
	print("\n1. Loading pretrained ViT model...")

	model_name = 'google/vit-base-patch16-224'
	processor = ViTImageProcessor.from_pretrained(model_name)

	# TRANSFORMS
	if config['augment']:  # FOR AUGMENTATION
		train_transform = transforms.Compose([
			transforms.Resize((config['resolution'], config['resolution'])),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomVerticalFlip(p=0.5),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(
       					brightness=0.2, 
                        contrast=0.2,
			),
			transforms.ToTensor(),
			transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((config['resolution'], config['resolution'])),
			transforms.ColorJitter(
       					brightness=0.2, 
                        contrast=0.2,
			),

			transforms.ToTensor(),
			transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
		])

	val_transform = transforms.Compose([
		transforms.Resize((config['resolution'], config['resolution'])),
		transforms.ToTensor(),
		transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
	])

	# CREATE SPLITS IF NOT EXIST
	if not os.path.exists('splits/train_split.csv'):
		create_data_splits(data_dir=DATA_DIR, seed=42)

	# DATASETS:
	trainset = TBDataset(csv_file='splits/train_split.csv', root_dir=DATA_DIR, transform=train_transform)
	valset = TBDataset(csv_file='splits/val_split.csv', root_dir=DATA_DIR, transform=val_transform)
	print(f'TRAIN : {len(trainset)}, VAL : {len(valset)}')

	trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
	valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)
	print(f"TRAINING SET: {len(trainset)} images")
	print(f"VALIDATION SET: {len(valset)} images")

	print(f"\n SETTING UP TRAINING...")

	model = ViTForImageClassification.from_pretrained(
		model_name,
		num_labels=2,                 # BINARY CLASSIFICATION
		ignore_mismatched_sizes=True  # Allow head replacement
	)
	model = model.to(device)
	print(f"VIT MODEL LOADED")

	# LOSS AND OPTIMIZER
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # Small learning rate for fine-tuning

	# TRAINING LOOP
	EPOCHS = 50
	print(f' EPOCHS : {EPOCHS}')

	print(' STARTING TRAINING... \n')

	# Create checkpoint directory
	os.makedirs('results/checkpoints', exist_ok=True)
	print('\n3. Starting training...')
	print('=' * 60)

	best_val_acc = 0.0
	best_epoch = 0
	start_time = time.time()  # Start timing

	for epoch in range(EPOCHS):
		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0

		print(f'EPOCH {epoch+1}/{EPOCHS}')
		print('-' * 60)

		for inputs, labels in trainloader:
			inputs, labels = inputs.to(device), labels.to(device)

			# Forward pass
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs.logits, labels)

			# Backward pass
			loss.backward()
			optimizer.step()

			# Statistics
			train_loss += loss.item()
			_, predicted = outputs.logits.max(1)
			train_total += labels.size(0)
			train_correct += (predicted == labels).sum().item()

		train_loss = train_loss / len(trainloader)
		train_accuracy = 100 * train_correct / train_total

		# EVALUATE ON VALIDATION SET
		print('EVALUATING :')
		model.eval()
		val_correct = 0
		val_total = 0
		val_loss = 0.0

		with torch.no_grad():
			for images, labels in valloader:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs.logits, labels)

				val_loss += loss.item()
				_, predicted = outputs.logits.max(1)
				val_total += labels.size(0)
				val_correct += (predicted == labels).sum().item()

		val_loss = val_loss / len(valloader)
		val_accuracy = 100 * val_correct / val_total

		# METRICS:
		print(f' Epoch [{epoch+1}/{EPOCHS}] ')
		print(f'TRAIN LOSS : {train_loss:.4f} | TRAIN ACC : {train_accuracy:.2f}%')
		print(f'VAL LOSS : {val_loss:.4f} | VAL ACC : {val_accuracy:.2f}%')

		# SAVE BEST MODEL
		if val_accuracy > best_val_acc:
			best_val_acc = val_accuracy
			best_epoch = epoch + 1
			model.save_pretrained('results/checkpoints/vit_best')
			processor.save_pretrained('results/checkpoints/vit_best_v2')
			print(f'BEST MODEL SAVED (ACCURACY : {best_val_acc:.2f}%)')
		print()

	# Calculate training time
	end_time = time.time()
	total_time = end_time - start_time
	avg_time_per_epoch = total_time / EPOCHS

	print('\n' + '-' * 60)
	print('TRAINING COMPLETE!')
	print('-' * 60)
	print(f'\n EXPERIMENT {EXPERIMENT} SUMMARY:')
	print('-' * 60)
	print(f"Model:                 {config['model'].upper()}")
	print(f"Image Resolution:     {config['resolution']}×{config['resolution']}")
	print(f"Pretrained Weights:   {'✅ Yes (ImageNet-21k)' if config['pretrained'] else '❌ No (random init)'}")
	print(f"Data Augmentation:    {'✅ Yes (flips, rotations, color jitter)' if config['augment'] else '❌ No (standard normalization only)'}")
	print('-' * 60)
	print(' RESULTS:')
	print('-' * 60)
	print(f"Total Training Time:  {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
	print(f"Avg Time per Epoch:   {avg_time_per_epoch:.2f} seconds")
	print(f"Best Val Accuracy:    {best_val_acc:.2f}% (Epoch {best_epoch}/{EPOCHS})")
	print('=' * 60)