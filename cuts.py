import cv2
import numpy as np
import os
import json

# === Directories and file paths ===
edge_input_dir = 'steak_edges_cleaned/'
mask_output_dir = 'steak_masks/'
cut_output_json = 'steak_approx_cuts.json'

os.makedirs(mask_output_dir, exist_ok=True)
cut_dict = {}

# === Process each edge image ===
for idx, edge_filename in enumerate(sorted(os.listdir(edge_input_dir))):
    edge_path = os.path.join(edge_input_dir, edge_filename)
    image = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

    # === Create mask from outer contours ===
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # === Save mask ===
    filename_no_ext, ext = os.path.splitext(edge_filename)
    base_name = filename_no_ext.split('_')[0]
    mask_filename = f"{base_name}_mask{ext}"
    mask_path = os.path.join(mask_output_dir, mask_filename)
    cv2.imwrite(mask_path, mask)
    print(f"Saved mask: {mask_path}")

    # === Compute vertical 50/50 cut ===
    height, width = mask.shape
    total_area = cv2.countNonZero(mask)
    target_area = total_area / 2

    def left_area(x): return cv2.countNonZero(mask[:, :x])
    x = width // 2
    if left_area(x) < target_area:
        while x < width and left_area(x) < target_area:
            x += 1
    else:
        while x > 0 and left_area(x) > target_area:
            x -= 1

    cut_dict[idx + 1] = x

# === Save cuts to JSON ===
with open(cut_output_json, 'w') as f:
    json.dump(cut_dict, f, indent=2)
print(f"Saved cut positions to {cut_output_json}")
