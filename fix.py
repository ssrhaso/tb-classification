import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

# ====== CONFIGURATION ======
NEG_DIR = 'src/datasets/tb_dataset_crops/neg'
POS_DIR = 'src/datasets/tb_dataset_crops/pos'
LABELS_PATH = 'src/datasets/tb_dataset_crops/labels.csv'
FIG1_PATH = 'Figure 1.jpg'
FIG2_PATH = 'Figure 2.jpg'
CROP_SIZE = 128  # Match to your training config (128 or 224)
CROPS_PER_IMAGE = 999  # Choose between 500-1000 per image

# ====== REMOVE OLD CROPS ======
def clean_crops(folder, prefix, max_id):
    for fname in os.listdir(folder):
        if fname.startswith(prefix):
            try:
                num = int(fname.split('_')[1].split('.')[0])
                if num > max_id:
                    os.remove(os.path.join(folder, fname))
            except Exception:
                continue

clean_crops(NEG_DIR, 'neg', 1000)
clean_crops(POS_DIR, 'pos', 1000)

# ====== CROP FUNCTION (AVOID BLACKSPACE) ======
def get_crops(img_path, outdir, prefix, count, start_idx):
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not open image at: {img_path}")
        exit(1)

    h, w = img.shape[:2]
    crops = 0
    stride = CROP_SIZE // 2
    filenames = []

    for y in tqdm(range(0, h - CROP_SIZE + 1, stride)):
        for x in range(0, w - CROP_SIZE + 1, stride):
            crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            nonblack_ratio = np.mean(np.any(crop > 30, axis=2))
            if nonblack_ratio < 0.8:
                continue
            fname = f"{prefix}_{start_idx + crops:04d}.jpg"
            cv2.imwrite(os.path.join(outdir, fname), crop)
            filenames.append(fname)
            crops += 1
            if crops >= count:
                return filenames
    return filenames

# ====== MAIN SCRIPT ======
if __name__ == '__main__':
    os.makedirs(POS_DIR, exist_ok=True)
    os.makedirs(NEG_DIR, exist_ok=True)

    # Start after existing crops
    neg_start = 1001
    pos_start = 1001

    print("Cropping negatives from Figure 1 (sparse)...")
    neg_files = get_crops(FIG1_PATH, NEG_DIR, 'neg', CROPS_PER_IMAGE, neg_start)

    print("Cropping positives from Figure 2 (bacteria dense)...")
    pos_files = get_crops(FIG2_PATH, POS_DIR, 'pos', CROPS_PER_IMAGE, pos_start)

    # ====== UPDATE LABELS CSV ======
    labels = []
    # Existing crops <= 1000
    for fname in sorted(os.listdir(POS_DIR)):
        if fname.startswith('pos_') and fname.endswith('.png'):
                labels.append([fname, 1])
    for fname in sorted(os.listdir(NEG_DIR)):
        if fname.startswith('neg_') and fname.endswith('.png'):
                labels.append([fname, 0])

    # Write CSV
    with open(LABELS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        writer.writerows(labels)

    print(f"Done! Negatives added: {len(neg_files)}, Positives added: {len(pos_files)}")
    print(f"Labels CSV updated: {LABELS_PATH} ({len(labels)} lines)")
