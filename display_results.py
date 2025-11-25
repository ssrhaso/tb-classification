import pandas as pd

# List of experiments
experiments = [
    {'id': 1, 'model': 'resnet18', 'resolution': 224, 'pretrained': True, 'augment': True},
    {'id': 2, 'model': 'resnet18', 'resolution': 224, 'pretrained': True, 'augment': False},
    {'id': 3, 'model': 'resnet18', 'resolution': 128, 'pretrained': True, 'augment': True},
    {'id': 4, 'model': 'resnet18', 'resolution': 128, 'pretrained': True, 'augment': False},
    {'id': 5, 'model': 'resnet18', 'resolution': 64, 'pretrained': True, 'augment': True},
    {'id': 6, 'model': 'resnet18', 'resolution': 64, 'pretrained': True, 'augment': False},
    {'id': 7, 'model': 'resnet18', 'resolution': 32, 'pretrained': True, 'augment': True},
    {'id': 8, 'model': 'resnet18', 'resolution': 32, 'pretrained': True, 'augment': False},
]

# For each experiment, input the results for each image tested
# For example: results[experiment_id] = [(correct, confidence), ... for 6 images]
results = {
    1: [(1,0.98), (1,0.92), (0,0.65), (1,0.93), (1,0.90), (0,0.78)],
    2: [(1,0.89), (1,0.94), (1,0.82), (0,0.60), (1,0.95), (0,0.70)],
    # Add results for each experiment id...
    8: [(1,0.74), (0,0.50), (0,0.68), (1,0.80), (0,0.62), (1,0.77)],
}

experiment_rows = []

for exp in experiments:
    exp_id = exp['id']
    img_results = results.get(exp_id, [])
    correct_sum = sum([c for c, conf in img_results])
    percent_correct = 100 * correct_sum / len(img_results) if img_results else 0
    avg_confidence = sum([conf for c, conf in img_results]) / len(img_results) if img_results else 0
    # Individual image scores as string
    individual_scores = ", ".join([f"img{i+1}:{c}|{conf:.2f}" for i, (c,conf) in enumerate(img_results)])
    experiment_rows.append({
        'id': exp_id,
        'model': exp['model'],
        'resolution': exp['resolution'],
        'pretrained': exp['pretrained'],
        'augment': exp['augment'],
        'percent_correct': f"{percent_correct:.1f}%",
        'avg_confidence': f"{avg_confidence:.2f}",
        'image_scores': individual_scores
    })

df = pd.DataFrame(experiment_rows)
print(df.to_string(index=False))
df.to_csv('results/experiment_summary.csv', index=False)