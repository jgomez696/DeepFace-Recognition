import cv2
import requests
import numpy as np
from io import BytesIO

def display_detected_face_from_url(image_url):
    try:
        # Download image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image from URL: {image_url}")
            return
        
        # Convert downloaded image to numpy array
        img_np = np.frombuffer(response.content, np.uint8)
        
        # Decode numpy array as an image
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Failed to decode image from URL: {image_url}")
            return

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained face detection model (you can use your preferred model here)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"No face detected in the image from URL: {image_url}")
            return

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

        # Display the image with detected faces
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        cv2.destroyAllWindows()

# Example usage with the provided image URL
image_url = 'https://i.pinimg.com/736x/a3/17/da/a317dafdb9a3ff791dd55193c18b11ce.jpg'
display_detected_face_from_url(image_url)
