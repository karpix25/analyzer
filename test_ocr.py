import os
import sys
import ssl

import cv2
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing import get_text_bottom_from_contours, refine_crop_rect

def test_image(image_path):
    print(f"Testing image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return

    H, W = frame.shape[:2]
    print(f"Image size: {W}x{H}")

    # Run the new logic
    text_bottom, valid_contours = get_text_bottom_from_contours(frame)

    print(f"Result text_bottom: {text_bottom}")
    print(f"Found {len(valid_contours)} valid text regions")

    # Visualize
    debug_img = frame.copy()
    for x, y, w, h in valid_contours:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if text_bottom:
        cv2.line(debug_img, (0, text_bottom), (W, text_bottom), (0, 0, 255), 4)
        
        # Test Auto-Crop
        margin = max(int(0.01 * H), 10)
        crop_top = text_bottom + margin
        crop_height = H - crop_top
        
        # Initial crop box
        x, y, w, h = 0, crop_top, W, crop_height
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red = Initial
        
        rx, ry, rw, rh = refine_crop_rect(frame, x, y, w, h)
        
        cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 3) # Yellow = Refined
        print(f"Auto-Crop: {w}x{h} -> {rw}x{rh}")

    output_path = "test_result.jpg"
    cv2.imwrite(output_path, debug_img)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Path to the uploaded image
    # Path to the uploaded image
    img_path = "/Users/nadaraya/.gemini/antigravity/brain/bcbd27f8-c7af-447d-a302-7780e5a73777/uploaded_image_1764203496194.jpg"
    test_image(img_path)
    test_image(img_path)
