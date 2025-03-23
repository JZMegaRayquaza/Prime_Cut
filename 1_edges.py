import cv2
import os
import numpy as np

# Input and output directories
input_dir = 'steaks/'
output_dir = 'steak_edges/'
os.makedirs(output_dir, exist_ok=True)

# Crop region
x, y, w, h = 75, 225, 500, 225

for steak in os.listdir(input_dir):
    input_path = os.path.join(input_dir, steak)
    image = cv2.imread(input_path)

    # Crop the image to focus on the steak
    cropped = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce texture noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Invert the edges to make edges white, background black
    edge_output = np.zeros_like(edges)
    edge_output[edges > 0] = 255

    # Get filename without extension
    filename, ext = os.path.splitext(steak)

    # Save edges image
    output_path = os.path.join(output_dir, f'{filename}_edges{ext}')
    cv2.imwrite(output_path, edge_output)

    print(f'Saved: {output_path}')

print('Edges complete!')
