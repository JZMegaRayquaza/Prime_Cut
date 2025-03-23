import cv2
import numpy as np
import os

# Input and output directories
input_dir = 'steak_edges_cleaned/'
output_dir = 'steak_masks/'
os.makedirs(output_dir, exist_ok=True)

# For each steak outline, determine mask
for steak_edges in os.listdir(input_dir):
    input_path = os.path.join(input_dir, steak_edges)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary threshold and find outermost contours
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill in detected contours with white
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Get filename without extension
    filename, ext = os.path.splitext(steak_edges)
    steak = filename.split('_')[0]

    # Save mask image
    output_path = os.path.join(output_dir, f'{steak}_mask{ext}')
    cv2.imwrite(output_path, mask)

    print(f'Saved: {output_path}')

print('Masks complete!')
