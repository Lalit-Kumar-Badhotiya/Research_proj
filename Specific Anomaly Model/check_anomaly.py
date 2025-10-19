import torch
from PIL import Image
from anodet.utils import standard_image_transform

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

# Get the threshold from the test script
# This is a simplified way to get a threshold. 
# A more robust way would be to calculate it from a validation set.
from anodet.test import optimal_threshold
from anodet import AnodetDataset
from torch.utils.data import DataLoader

test_dataset = AnodetDataset(image_directory_path='data/test/all')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, image_classifications_target, masks_target, image_scores, score_maps = model.evaluate(test_dataloader)

precision, recall, threshold = optimal_threshold(image_classifications_target, image_scores)

print(f"Image: {image_path}")
print(f"Anomaly Score: {image_score.item()}")
print(f"Anomaly Threshold: {threshold}")

if image_score.item() > threshold:
    print("Result: Anomaly detected")
else:
    print("Result: No anomaly detected")
