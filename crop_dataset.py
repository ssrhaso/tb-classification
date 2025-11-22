import os
import cv2
import numpy as np
import csv

# Paths
NEG_DIR = 'src/datasets/tb_dataset_crops/neg'
POS_DIR = 'src/datasets/tb_dataset_crops/pos'
LABELS_PATH = 'src/datasets/tb_dataset_crops/labels.csv'
FIG1_PATH = 'Figure 1.jpg'   # sparse/tb-negative
FIG2_PATH = 'Figure 2.jpg'   # dense/tb-positive

# Cropping configs
LARGE_SIZE = 1024
LARGE_STRIDE = 512
SMALL_SIZE = 512
SMALL_STRIDE = 256
NEW_LARGE_START = 1000            # Large crops use index >= 1000
NEW_SMALL_START = 2000            # Small crops use index >= 2000
CROPS_PER_IMAGE_LARGE = 20        # Number of large crops per image
CROPS_PER_IMAGE_SMALL = 40        # Number of small crops per image

os.makedirs(NEG_DIR, exist_ok=True)
os.makedirs(POS_DIR, exist_ok=True)

# Delete previous large and small crops (≥1000 and ≥2000)
def clean_new(folder, prefix, cutoff):
    for fname in os.listdir(folder):
        if fname.startswith(prefix) and fname.endswith('.jpg'):
            try:
                idx = int(fname.split('_')[1].split('.')[0])
                if idx >= cutoff:
                    os.remove(os.path.join(folder, fname))
                    print(f"Deleted {fname}")
            except Exception as e:
                print(f"Error deleting {fname}: {e}")

print("Deleting ALL previous new crops...")
clean_new(NEG_DIR, 'neg', NEW_LARGE_START)
clean_new(POS_DIR, 'pos', NEW_LARGE_START)
clean_new(NEG_DIR, 'neg', NEW_SMALL_START)
clean_new(POS_DIR, 'pos', NEW_SMALL_START)

# Cropping function (filtered for black edges)
def crop_and_save(img_path, out_dir, prefix, start_idx, crop_size, stride, max_crops):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    crops = 0
    filenames = []
    for y in range(0, h-crop_size+1, stride):
        for x in range(0, w-crop_size+1, stride):
            crop = img[y:y+crop_size, x:x+crop_size]
            black_mask = np.all(crop < 40, axis=2)
            black_ratio = np.mean(black_mask)
            if black_ratio > 0.2:
                continue
            fname = f"{prefix}_{start_idx+crops:04d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), crop)
            filenames.append((fname, 1 if prefix=='pos' else 0))
            crops += 1
            if crops >= max_crops:
                break
        if crops >= max_crops:
            break
    print(f"Saved {crops} crops ({crop_size}px) for {prefix} from {img_path}")
    return filenames

# Make LARGE (1024px) crops
print("\nCropping LARGE negatives from Figure 1...")
neg_files_large = crop_and_save(FIG1_PATH, NEG_DIR, 'neg', NEW_LARGE_START, LARGE_SIZE, LARGE_STRIDE, CROPS_PER_IMAGE_LARGE)
print("\nCropping LARGE positives from Figure 2...")
pos_files_large = crop_and_save(FIG2_PATH, POS_DIR, 'pos', NEW_LARGE_START, LARGE_SIZE, LARGE_STRIDE, CROPS_PER_IMAGE_LARGE)

# Make SMALL (512px) crops
print("\nCropping SMALL negatives from Figure 1...")
neg_files_small = crop_and_save(FIG1_PATH, NEG_DIR, 'neg', NEW_SMALL_START, SMALL_SIZE, SMALL_STRIDE, CROPS_PER_IMAGE_SMALL)
print("\nCropping SMALL positives from Figure 2...")
pos_files_small = crop_and_save(FIG2_PATH, POS_DIR, 'pos', NEW_SMALL_START, SMALL_SIZE, SMALL_STRIDE, CROPS_PER_IMAGE_SMALL)

# Build new labels.csv (all old <1000, plus large and small new crops)
data = []
for f in os.listdir(POS_DIR):
    try:
        idx = int(f.split('_')[1].split('.')[0])
        if idx < NEW_LARGE_START:
            data.append([f, 1])
    except Exception:
        continue
for f in os.listdir(NEG_DIR):
    try:
        idx = int(f.split('_')[1].split('.')[0])
        if idx < NEW_LARGE_START:
            data.append([f, 0])
    except Exception:
        continue

data += pos_files_large + neg_files_large + pos_files_small + neg_files_small

with open(LABELS_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filepath', 'label'])
    writer.writerows(data)

print("\nDone. All new crops created (large+small), old ones deleted, blackspace ignored, labels updated!")
