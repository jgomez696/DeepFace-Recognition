import cv2
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

# Initialize MTCNN and InceptionResnetV1 models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to download and preprocess images from URLs
def preprocess_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image from URL: {image_url}")
            return None
        
        # Convert image to PIL Image
        img = Image.open(BytesIO(response.content))
        
        # Convert PIL Image to numpy array and BGR format (for OpenCV compatibility)
        img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        return img_rgb

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {str(e)}")
        return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

# Function to recognize faces using facenet-pytorch
def recognize_faces(url1, url2):
    # Preprocess images from URLs
    img1 = preprocess_image_from_url(url1)
    img2 = preprocess_image_from_url(url2)

    if img1 is None or img2 is None:
        return

    # Detect faces and get embeddings
    def get_embeddings(img):
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                face = Image.fromarray(face)
                face_tensor = transforms.functional.to_tensor(face).unsqueeze(0).to(device)
                faces.append(face_tensor)
            if faces:
                embeddings = resnet(torch.cat(faces)).detach().cpu().numpy()
                return embeddings
        return None

    embeddings1 = get_embeddings(img1)
    embeddings2 = get_embeddings(img2)

    if embeddings1 is None or embeddings2 is None:
        print("No faces detected in one of the images.")
        return

    # Compare embeddings to recognize faces
    distance = np.linalg.norm(embeddings1[0] - embeddings2[0])
    print(f"Distance between faces: {distance}")
    if distance < 0.7:
        match_result = "Faces are a match."
    else:
        match_result = "Faces are not a match."

    # Plot images and match result
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Image 1')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Image 2')
    ax[1].axis('off')

    fig.suptitle(match_result, fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage with the provided image URLs
url1 = 'https://i.pinimg.com/736x/ef/b2/b7/efb2b77b15bd1de56c0b42845e9013cf.jpg'
url2 = 'https://i.pinimg.com/564x/94/48/78/944878cfa49fe275e13c8e0ae0c1f129.jpg'
recognize_faces(url1, url2)
