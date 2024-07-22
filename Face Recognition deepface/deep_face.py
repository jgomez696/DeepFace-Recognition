from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import os

# Function to download and save image from URL
def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            return True
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception occurred while downloading image from {url}: {str(e)}")
        return False

# URLs of the images
url1 = 'https://i.pinimg.com/736x/ef/b2/b7/efb2b77b15bd1de56c0b42845e9013cf.jpg'
url2 = 'https://i.pinimg.com/736x/62/54/de/6254dec95b842ded0f109f478c68d378.jpg'

# File paths to save images locally
img_dir = 'img'
img1_path = os.path.join(img_dir, 'img_from_url1.jpg')
img2_path = os.path.join(img_dir, 'img_from_url2.jpg')

# Download images from URLs
success1 = download_image(url1, img1_path)
success2 = download_image(url2, img2_path)

if success1 and success2:
    # Function to verify if images are of the same person
    def verify(img1_path, img2_path):
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None:
                print(f"Failed to read image from {img1_path}")
                return
            if img2 is None:
                print(f"Failed to read image from {img2_path}")
                return
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Image 1')
            axs[0].axis('off')
            
            axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axs[1].set_title('Image 2')
            axs[1].axis('off')
            
            # Verify similarity
            output = DeepFace.verify(img1_path, img2_path)
            verification = output['verified']
            
            if verification:
                axs[0].text(10, 10, 'Same Person', color='green', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
                axs[1].text(10, 10, 'Same Person', color='green', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            else:
                axs[0].text(10, 10, 'Different Person', color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
                axs[1].text(10, 10, 'Different Person', color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Exception occurred during verification: {str(e)}")

    # Verify if images are of the same person
    verify(img1_path, img2_path)
else:
    print("Failed to download one or both of the images. Check the URLs and try again.")
