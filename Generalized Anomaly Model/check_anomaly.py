import torch
from PIL import Image
from anodet.utils import standard_image_transform
import os
import numpy as np

# Load the model
model = torch.load('padim_model.pth', weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to_device(device)

# Load and transform the image
image_path = 'data/test/all/0.png'
image = Image.open(image_path).convert('RGB')
transformed_image = standard_image_transform(image)
batch = transformed_image.unsqueeze(0).to(device)

# Predict
image_score, score_map = model.predict(batch)

# Get the threshold
threshold_file = 'threshold.txt'
if os.path.exists(threshold_file):
    with open(threshold_file, 'r') as f:
        threshold = float(f.read())
else:
    from anodet.test import optimal_threshold
    from anodet import AnodetDataset
    from torch.utils.data import DataLoader

    print("Calculating threshold...")
    test_dataset = AnodetDataset(image_directory_path='data/train/good') # Use good images for threshold
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    images, image_classifications_target, masks_target, image_scores, score_maps = model.evaluate(test_dataloader)

    precision, recall, threshold = optimal_threshold(image_classifications_target, image_scores)
    with open(threshold_file, 'w') as f:
        f.write(str(threshold))

print(f"Image: {image_path}")
print(f"Anomaly Score: {image_score.item()}")
print(f"Anomaly Threshold: {threshold}")

if image_score.item() > threshold:
    print("Result: Anomaly detected")
else:
    print("Result: No anomaly detected")