"""Rename patches to match dataset format: pos_1000.jpg, pos_1001.jpg, etc."""
from pathlib import Path

crop_dir = Path('datasets/new_crops/pos/')
patches = sorted(crop_dir.glob('*.jpg'))

print(f"Found {len(patches)} patches")
print("Renaming to: pos_1000.jpg, pos_1001.jpg, ...")

# Find starting number (check existing training data)
train_dir = Path('datasets/tb_dataset_crops/pos/')
if train_dir.exists():
    existing = list(train_dir.glob('pos_*.jpg'))
    if existing:
        # Get max number from existing files
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except:
                pass
        start_num = max(numbers) + 1 if numbers else 1000
    else:
        start_num = 1000
else:
    start_num = 1000

print(f"Starting from: pos_{start_num}.jpg")

# Rename
for i, old_path in enumerate(patches):
    new_name = f"pos_{start_num + i}.jpg"
    new_path = crop_dir / new_name
    old_path.rename(new_path)

print(f"✓ Renamed {len(patches)} files")
print(f"Range: pos_{start_num}.jpg to pos_{start_num + len(patches) - 1}.jpg")
print("\nNEXT: Run sample script (keep 150), then merge")
