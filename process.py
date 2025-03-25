import cv2
import os
import numpy as np

# Input and output directories
input_dir = 'steaks/'
crop_dir = 'steak_crops/'
blur_dir = 'steak_blurs/'
edge_dir = 'steak_edges/'

# Create output directories if they don't exist
os.makedirs(crop_dir, exist_ok=True)
os.makedirs(blur_dir, exist_ok=True)
os.makedirs(edge_dir, exist_ok=True)

# Crop region
x, y, w, h = 75, 225, 500, 225

for steak in os.listdir(input_dir):
    input_path = os.path.join(input_dir, steak)
    image = cv2.imread(input_path)

    # Crop the image to focus on the steak
    cropped = image[y:y+h, x:x+w]
    
    # Save cropped image
    crop_path = os.path.join(crop_dir, steak)
    cv2.imwrite(crop_path, cropped)

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce texture noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Save blurred grayscale image
    blur_path = os.path.join(blur_dir, steak)
    cv2.imwrite(blur_path, blurred)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Invert the edges to make edges white, background black
    edge_output = np.zeros_like(edges)
    edge_output[edges > 0] = 255

    # Save edges image
    edge_path = os.path.join(edge_dir, steak)
    cv2.imwrite(edge_path, edge_output)

    print(f'Saved: {crop_path}, {blur_path}, {edge_path}')

print('Processing complete!')
