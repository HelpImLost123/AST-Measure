import os

import cv2
import numpy as np

def read_images(file_path):
    # Read the image from the specified file path
    image = cv2.imread(file_path)
    
    # Check if the image was successfully read
    if image is None:
        print(f"Error: Could not read the image from {file_path}")
        return None
    
    return image

def write_image(file_path, image):
    # Write the image to the specified file path, if the directory don't exist create it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    success = cv2.imwrite(file_path, image)
    
    if not success:
        print(f"Error: Could not write the image to {file_path}")
    else:
        print(f"Image successfully written to {file_path}")
        

# seSize: structuring element size for morphological operations, if None it will be calculated based on the minimum pellet size.
# min_pellet: the estimated number of pellets of the smaller estimated size that can fit on the petri dish
# max_pellet: the estimated number of pellets of the bigger estimated size that can fit on the petri dish
def find_pellets(image, intensity_percentage=0.95, seSize=None, min_pellet=30, max_pellet=4, compactness_threshold=0.75, retry_step=0.05, output_path=None):
    image = image.copy()
    # Check if the image has 3 channels (not grayscale)
    if len(image.shape) == 3 and image.shape[2] == 3:
        blue_channel = image[:, :, 0]
        gray = blue_channel
    else:
        gray = image
    # cv2.imshow('Blue Channel', gray)
    

    equalized = cv2.equalizeHist(gray)
    # cv2.imshow('Equalized Image', equalized)
    
    minPelletDiameter = int((min(gray.shape) / min_pellet))
    maxPelletDiameter = int((min(gray.shape) / max_pellet))
    
    if seSize is None:
        # estimate the size to be smaller than the minimum pellet size to avoid merging multiple pellets together
        # ,but not too small to avoid being affected by noise.
        seSize = max(3, int(minPelletDiameter / 5))
        
    smallerSE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    biggerSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seSize, seSize))

    img_h, img_w = image.shape[:2]
    threshold = intensity_percentage
    
    detected = []

    while threshold >= 0:
        print(f"Structuring Element Size: {seSize}")
        print(f"intensity threshold: {threshold:.2f}")

        _, binary = cv2.threshold(equalized, 255 * threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Binary Image', binary)
        
        # smaller closing to fill small holes of the text labels first
        filled_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smallerSE)
        opened_image = cv2.morphologyEx(filled_binary, cv2.MORPH_OPEN, biggerSE)
        
        # cv2.imshow('Opened Image', opened_image)
        
        if output_path:
            write_image(f'{output_path}\\1_gray.jpg', gray)
            write_image(f'{output_path}\\2_equalized.jpg', equalized)
            write_image(f'{output_path}\\3_binary.jpg', binary)
            write_image(f'{output_path}\\4_opened.jpg', opened_image)
        
        contours = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area <= 0:
                continue
            
            # Draw the contour on a tiny blank canvas
            x, y, w, h = cv2.boundingRect(contour)
            local_mask = np.zeros((h, w), dtype=np.uint8)
            
            shifted_contour = contour - [x, y]
            cv2.drawContours(local_mask, [shifted_contour], -1, 255, -1)
            
            # Find the deepest point ignoring outward spikes
            dist_transform = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            cx = max_loc[0] + x
            cy = max_loc[1] + y
            radius = max_val
            diameter = radius * 2

            # Filter out the objects that are too small or too large to be pellets
            if diameter < minPelletDiameter or diameter > maxPelletDiameter:
                continue

            # Filter out the circle that are out of bounds
            if cx - radius < 0 or cy - radius < 0 or cx + radius > img_w or cy + radius > img_h:
                continue
            
            # Calculate the area of the distance transform circle
            inscribed_area = np.pi * (radius ** 2)
            
            # Compare it to the contour's area
            compactness = inscribed_area / contour_area
            
            if compactness < compactness_threshold:
                continue

            detected.append((cv2.boundingRect(contour), cx, cy, diameter, compactness))

        if detected:
            for (x, y, w, h), cx, cy, diameter, compactness in detected:
                cv2.circle(image, (int(cx), int(cy)), int(diameter/2), (0, 255, 0), 2)
                print(f"Detected pellet | cx: {cx}, cy: {cy}, diameter: {diameter:.1f} px, compactness: {compactness:.2f}")
            if output_path:
                write_image(f'{output_path}\\5_detected.jpg', image)
            break

        if retry_step > 0:
            print(f"No pellets found at threshold {threshold:.2f}, retrying...")
            threshold -= retry_step
        else:
            print("No pellets detected after exhausting all thresholds.")
            break

    cv2.imshow('Detected Pellets', image)
    
    return detected

# Visualize the maximum and minimum pellet size on the image for easier parameter tuning.
def visualize_max_min_pellet(image, min_pellet=30, max_pellet=4):
    out = image.copy()
    img_h, img_w = out.shape[:2]
    shorter_dim = min(img_h, img_w)

    min_dia = max(2, int(shorter_dim / min_pellet))
    max_dia = max(2, int(shorter_dim / max_pellet))

    n_min = max(1, shorter_dim // min_dia)
    n_max = max(1, shorter_dim // max_dia)

    pad = 8
    label_h = 20

    row1_h = label_h + min_dia + pad
    row2_h = label_h + max_dia + pad
    strip_h = pad + row1_h + pad + row2_h + pad

    # Semi-transparent dark strip at the bottom of the image
    y_strip = img_h - strip_h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y_strip), (img_w, img_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # --- Min pellet row (green) ---
    y0 = y_strip + pad
    cv2.putText(out, f"Min pellet: {min_dia}px dia  (x{n_min} fit)",
                (pad, y0 + label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA)
    cy_min = y0 + label_h + min_dia // 2
    for i in range(n_min):
        cx = i * min_dia + min_dia // 2
        cv2.circle(out, (cx, cy_min), min_dia // 2, (60, 220, 60), 2, cv2.LINE_AA)

    # --- Max pellet row (blue) ---
    y1 = y_strip + pad + row1_h + pad
    cv2.putText(out, f"Max pellet: {max_dia}px dia  (x{n_max} fit)",
                (pad, y1 + label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 220), 1, cv2.LINE_AA)
    cy_max = y1 + label_h + max_dia // 2
    for i in range(n_max):
        cx = i * max_dia + max_dia // 2
        cv2.circle(out, (cx, cy_max), max_dia // 2, (80, 120, 220), 2, cv2.LINE_AA)

    return out

def main():
    file_name = '7.23.1. original.jpg'
    img = read_images(f'dataset\\{file_name}')

    if img is not None:
        detected = find_pellets(img, output_path=f'output\\{file_name}', seSize=7, intensity_percentage=0.9)

        # size_viz = visualize_max_min_pellet(img, min_pellet=20, max_pellet=12)
        # cv2.imshow('Min / Max Pellet Size', size_viz)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found or could not be read.")
        
if __name__ == "__main__":
    main()