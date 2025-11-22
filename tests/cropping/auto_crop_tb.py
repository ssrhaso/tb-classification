"""Auto-crop with better black space filtering"""
import cv2
import numpy as np
from pathlib import Path
import os

def auto_crop(image_path, output_dir, label='positive', patch_size=224, stride=112):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    print(f"Size: {w}x{h}, Patch: {patch_size}x{patch_size}")
    
    saved = skipped_dark = skipped_blackspace = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            # Filter 1: Skip very dark patches
            mean_intensity = np.mean(gray)
            if mean_intensity < 40:
                skipped_dark += 1
                continue
            
            # Filter 2: Check for black edges/corners (circular FOV)
            black_pixels = np.sum(gray < 20)
            total_pixels = patch_size * patch_size
            black_ratio = black_pixels / total_pixels
            
            if black_ratio > 0.15:
                skipped_blackspace += 1
                continue
            
            # Filter 3: Skip low contrast (uniform/empty patches)
            std_intensity = np.std(gray)
            if std_intensity < 15:
                skipped_dark += 1
                continue
            
            # Save patch
            filename = f"{label}_{image_name}_x{x}_y{y}.jpg"
            cv2.imwrite(str(output_dir / filename), patch)
            saved += 1
    
    print(f"✓ Saved: {saved}")
    print(f"✗ Skipped dark: {skipped_dark}")
    print(f"✗ Skipped blackspace: {skipped_blackspace}")
    return saved

if __name__ == '__main__':
    print("="*60)
    
    # Clean old crops - DELETE FILES ONLY, NOT FOLDER
    crop_dir = Path('datasets/new_crops/pos/')
    if crop_dir.exists():
        print("Cleaning old crops...")
        for old_file in crop_dir.glob('*.jpg'):
            try:
                os.remove(old_file)
            except:
                pass
        print(f"✓ Cleaned {len(list(crop_dir.glob('*.jpg')))} old files")
    
    total = 0
    total += auto_crop('Figure 1.jpg', 'datasets/new_crops/pos/', 'positive')
    total += auto_crop('Figure 2.jpg', 'datasets/new_crops/pos/', 'positive')
    
    print(f"\n{'='*60}")
    print(f"Total: {total} patches → datasets/new_crops/pos/")
    print(f"{'='*60}")
    
    if total > 0:
        print("\n✓ Much better filtering applied!")
        print("NEXT: Run merge script")
    else:
        print("\n⚠ No patches saved - filters might be too strict")
