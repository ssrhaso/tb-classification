"""
Merge new cropped patches into existing training dataset.
Copies files from datasets/new_crops/pos/ to datasets/tb_dataset_crops/pos/
"""

import shutil
from pathlib import Path

def merge_crops(source_dir, target_dir):
    """Copy new crops to training dataset"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return
    
    # Create target if doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Count before
    before_count = len(list(target_dir.glob('*.jpg')))
    new_files = list(source_dir.glob('*.jpg'))
    
    print("="*60)
    print("MERGING NEW CROPS INTO TRAINING SET")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Existing files in target: {before_count}")
    print(f"New files to copy: {len(new_files)}")
    
    if len(new_files) == 0:
        print("\n⚠ No files to merge!")
        return
    
    # Copy files
    copied = 0
    for src_file in new_files:
        dst_file = target_dir / src_file.name
        
        # Avoid overwriting - add suffix if file exists
        if dst_file.exists():
            counter = 1
            stem = src_file.stem
            while dst_file.exists():
                dst_file = target_dir / f"{stem}_copy{counter}.jpg"
                counter += 1
        
        shutil.copy(src_file, dst_file)
        copied += 1
    
    # Count after
    after_count = len(list(target_dir.glob('*.jpg')))
    
    print(f"\n✓ Merge complete!")
    print(f"Files copied: {copied}")
    print(f"Total files in training set now: {after_count} (+{after_count - before_count})")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Update train_cnn.py (add ColorJitter)")
    print("2. Update train_vit.py (add ColorJitter)")
    print("3. Retrain both models")
    print("4. Test on unseen data")
    print("="*60)


if __name__ == '__main__':
    # Merge positive crops
    merge_crops(
        source_dir='datasets/new_crops/pos/',
        target_dir='datasets/tb_dataset_crops/pos/'
    )
