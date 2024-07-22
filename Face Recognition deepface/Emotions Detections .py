from deepface import DeepFace
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import numpy as np

# URL to the image you want to analyze
image_url = 'https://i.pinimg.com/736x/ec/6b/76/ec6b76ce531990ccc6bad65d390fcb92.jpg'

try:
    # Download and open the image from URL
    with urllib.request.urlopen(image_url) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())
    
    # Analyzing the image for emotion detection
    result = DeepFace.analyze(img_path='temp.jpg', actions=['emotion'])
    
    # Check if result is a list and retrieve emotions from the first element
    if isinstance(result, list) and len(result) > 0:
        first_result = result[0]  # Assuming the first result contains the desired data
        if 'emotion' in first_result:
            emotions = first_result['emotion']
            print("Emotions detected:")
            for emotion, value in emotions.items():
                print(f"{emotion}: {value}")
                
            # Open image using PIL
            img = Image.open('temp.jpg')
            img = np.array(img)
            
            # Initialize matplotlib figure
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            
            # Add text annotations for each emotion detected
            for emotion, value in emotions.items():
                plt.text(10, 20 + list(emotions.keys()).index(emotion) * 20, f'{emotion}: {value:.2f}', fontsize=12, color='red', weight='bold')
            
            plt.axis('off')
            plt.title('Emotions Detected')
            plt.show()
            
        else:
            print("Emotion key not found in the first result:", first_result)
    else:
        print("No valid results found.")
        
except Exception as e:
    print(f"Error analyzing face: {e}")

# Clean up: Remove temporary image file
import os
os.remove('temp.jpg')
