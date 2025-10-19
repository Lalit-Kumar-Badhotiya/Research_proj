import torch
from torch.utils.data import DataLoader
from anodet import AnodetDataset, Padim

# 1. Prepare your data
# Create a directory with your normal training images.
# For example: 'data/train/good'
train_image_directory = 'data/train/good' 

# 2. Create a dataset and dataloader
train_dataset = AnodetDataset(image_directory_path=train_image_directory)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Padim(backbone='resnet18', device=device)
model.fit(train_dataloader)

# 4. Save the trained model
torch.save(model, 'padim_model.pth')

print("Training complete. Model saved to padim_model.pth")
