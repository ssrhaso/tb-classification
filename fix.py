import os
import pandas as pd

pos_dir = 'src/datasets/tb_dataset_crops/pos/'
neg_dir = 'src/datasets/tb_dataset_crops/neg/'

# Get all files
pos_files = sorted([f for f in os.listdir(pos_dir) if f.endswith(('.jpg', '.png'))])
neg_files = sorted([f for f in os.listdir(neg_dir) if f.endswith(('.jpg', '.png'))])

# Create dataframe
data = []
for f in pos_files:
    data.append({'filepath': f, 'label': 1})
for f in neg_files:
    data.append({'filepath': f, 'label': 0})

df = pd.DataFrame(data)
df.to_csv('src/datasets/tb_dataset_crops/labels.csv', index=False)

print('✅ Updated labels.csv!')
print(f'Total: {len(df)} samples')
print(f'Positive: {(df["label"]==1).sum()}')
print(f'Negative: {(df["label"]==0).sum()}')
