import torch
from torch.utils.data import DataLoader
from anodet import AnodetDataset, visualize_eval_data

# 1. Prepare your test data
# Create a directory with your test images.
# This directory can contain both normal and anomalous images.
test_image_directory = 'data/test/all'

# 2. Load the trained model
model = torch.load('padim_model.pth', weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to_device(device)

# 3. Create a dataset and dataloader for testing
# For evaluation, you can also provide masks for anomalous regions if you have them.
test_dataset = AnodetDataset(image_directory_path=test_image_directory)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Evaluate the model
images, image_classifications_target, masks_target, image_scores, score_maps = model.evaluate(test_dataloader)

# 5. Visualize the results
# This will print ROC-AUC scores and show plots for image and pixel-level performance.
visualize_eval_data(image_classifications_target, masks_target, image_scores, score_maps)
