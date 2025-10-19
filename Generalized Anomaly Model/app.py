
import os
import torch
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from anodet.utils import standard_image_transform
from anodet.visualization import heatmap_image, boundary_image
import cv2

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('padim_model.pth', map_location=device, weights_only=False)
model.to_device(device)

@app.route('/')
def index():
    image_names = os.listdir('data/test/all')
    return render_template('index.html', image_names=image_names)

@app.route('/analyze', methods=['POST'])
def analyze():
    image_name = request.form['image_name']
    image_path = os.path.join('data/test/all', image_name)

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = standard_image_transform(image)
    batch = transformed_image.unsqueeze(0).to(device)

    # Predict
    image_score, score_map = model.predict(batch)

    # Generate heatmap
    #heatmap = heatmap_image(np.array(image), score_map.squeeze().cpu().numpy())

    # Generate boundary image
    # We need a threshold to create a binary mask for the boundary image.
    # For simplicity, we'll use a static threshold here.
    # A more robust approach would be to determine this threshold from a validation set.
    threshold = 10.0 
    binary_mask = (score_map > threshold).squeeze().cpu().numpy()
    boundary = boundary_image(np.array(image), binary_mask)
    
    # Save the images to the static folder
    original_path = os.path.join('static', 'original.png')
    #heatmap_path = os.path.join('static', 'heatmap.png')
    boundary_path = os.path.join('static', 'boundary.png')

    Image.fromarray(np.array(image)).save(original_path)
    #Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray(boundary).save(boundary_path)

    return render_template('result.html', 
                           original_image=original_path, 
                           #heatmap_image=heatmap_path, 
                           boundary_image=boundary_path,
                           score=image_score.item())

if __name__ == '__main__':
    app.run(debug=True)
