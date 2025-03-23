import cv2
import numpy as np
import os
import json

input_dir = 'steak_masks/'
output_json = 'steak_approx_cuts.json'

cut_dict = {}

# For each steak mask, determine 50/50 cut (x-value)
for i, steak_mask in enumerate(os.listdir(input_dir)):
    mask_path = os.path.join(input_dir, steak_mask)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Get target area of steak
    height, width = mask.shape
    total_area = cv2.countNonZero(mask)
    target_area = total_area / 2

    def left_area(x):
        return cv2.countNonZero(mask[:, :x])

    x = width // 2

    # Left cut is smaller than right cut
    if left_area(x) < target_area:
        while left_area(x) < target_area:
            x += 1
    # Left cut is larger than right cut
    else:
        while left_area(x) > target_area:
            x -= 1

    cut_dict[i+1] = x

# Save the x-values for each image to a JSON file
with open(output_json, 'w') as f:
    json.dump(cut_dict, f, indent=2)

print(f'Saved cut positions to {output_json}')
