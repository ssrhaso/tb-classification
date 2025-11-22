"""
Update labels.csv to include new files.
Format: filepath, label
"""

from pathlib import Path
import pandas as pd

def update_labels():
    """Add new files to labels.csv"""
    
    # Get all files in training folder
    train_dir = Path('src/datasets/tb_dataset_crops/pos/')
    all_files = sorted([f.name for f in train_dir.glob('pos_*.jpg')])
    print(f"Total files in pos/ folder: {len(all_files)}")
    
    # Filter to only files >= pos_1000 (new ones)
    new_files = [f for f in all_files if int(f.split('_')[1].split('.')[0]) >= 1000]
    
    if len(new_files) == 0:
        print("\n⚠ No new files found")
        return 0
    
    print(f"Found {len(new_files)} new files: {new_files[0]} to {new_files[-1]}")
    
    # Read existing labels.csv
    labels_csv = Path('src/datasets/tb_dataset_crops/labels.csv')
    df = pd.read_csv(labels_csv)
    print(f"Existing labels.csv: {len(df)} files")
    
    # Get existing filepaths
    existing = set(df['filepath'].tolist())
    
    # Add new files
    new_rows = []
    for filename in new_files:
        if filename not in existing:
            new_rows.append({
                'filepath': filename,
                'label': 1  # positive
            })
    
    if len(new_rows) == 0:
        print("✓ All files already in labels.csv")
        return 0
    
    # Append and save
    df_new = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(labels_csv, index=False)
    
    print(f"✓ Updated labels.csv: {len(df)} files (+{len(new_rows)})")
    return len(new_rows)

if __name__ == '__main__':
    print("="*60)
    print("UPDATING labels.csv")
    print("="*60)
    
    added = update_labels()
    
    print("\n" + "="*60)
    if added > 0:
        print("✓ Done! Now add ColorJitter and retrain")
    else:
        print("✓ Already up to date")
    print("="*60)
