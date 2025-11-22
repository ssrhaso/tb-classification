"""
TB Dataset Loader
Creates train/val/test splits and handles transforms
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class TBDataset(Dataset):
    """Tuberculosis Binary Classification Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to labels CSV (e.g., 'splits/train_split.csv')
            root_dir (str): Directory with pos/ and neg/ folders
            transform: torchvision transforms to apply
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get image filename and label from CSV
        img_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        
        # Construct path based on label (pos or neg folder)
        folder = 'pos' if label == 1 else 'neg'
        img_path = os.path.join(self.root_dir, folder, img_name)
            
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_data_splits(data_dir, seed=42):
    """
    Create train/val/test splits from labels.csv
    Saves to splits/train_split.csv, splits/val_split.csv, splits/test_split.csv
    
    Args:
        data_dir: Path to TB dataset (should contain labels.csv)
        seed: Random seed for reproducibility
    """
    labels_path = os.path.join(data_dir, 'labels.csv')
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    
    df = pd.read_csv(labels_path)
    
    # Split: 70% train, 15% val, 15% test (stratified by label)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=seed, 
        stratify=df.iloc[:, 1]
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=seed, 
        stratify=temp_df.iloc[:, 1]
    )
    
    # SAVE SPLITS
    os.makedirs('splits', exist_ok=True)
    train_df.to_csv('splits/train_split.csv', index=False)
    val_df.to_csv('splits/val_split.csv', index=False)
    test_df.to_csv('splits/test_split.csv', index=False)
    
    print(f"Data splits created:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df
