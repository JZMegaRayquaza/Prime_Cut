import cv2
import os
import numpy as np

# Input and output directories
input_dir = 'steaks/'
output_dir = 'steak_crops/'
os.makedirs(output_dir, exist_ok=True)

# Crop region
x, y, w, h = 75, 225, 500, 225

for steak in os.listdir(input_dir):
    input_path = os.path.join(input_dir, steak)
    image = cv2.imread(input_path)

    # Crop the image to focus on the steak
    cropped = image[y:y+h, x:x+w]

    # Get filename without extension
    filename, ext = os.path.splitext(steak)

    # Save cropped image
    output_path = os.path.join(output_dir, f'{filename}_crop{ext}')
    cv2.imwrite(output_path, cropped)

    print(f'Saved: {output_path}')

print('Crops complete!')