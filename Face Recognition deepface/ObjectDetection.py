import torch
from transformers import DetrForObjectDetection
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# URL to the image you want to analyze
image_url = 'https://i.pinimg.com/564x/41/b3/4b/41b34b63eeffdc9d630165ff8bfe6c6f.jpg'

# Download the image
image = Image.open(requests.get(image_url, stream=True).raw)

# Convert image to PyTorch tensor
image_tensor = torch.tensor(np.array(image))

# Load DETR model for object detection
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Perform object detection
outputs = model(image_tensor)

# Define categories for living and non-living objects (modify as needed)
living_categories = ['person', 'animal']
non_living_categories = ['car', 'chair', 'table', 'bottle', 'book']

# Initialize counters
living_count = 0
non_living_count = 0

# Iterate over detected objects and classify them
for pred in outputs['predictions']:
    if pred['label'] in living_categories:
        living_count += 1
    elif pred['label'] in non_living_categories:
        non_living_count += 1

# Plotting the results
categories = ['Living', 'Non-Living']
counts = [living_count, non_living_count]

plt.bar(categories, counts, color=['green', 'blue'])
plt.xlabel('Object Category')
plt.ylabel('Count')
plt.title('Living vs. Non-Living Objects Detected')
plt.show()
