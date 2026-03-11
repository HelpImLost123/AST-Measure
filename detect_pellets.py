import math
import os

import cv2
import numpy as np

from canny2 import filter_rgb_to_black, manual_canny

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
def find_pellets(image, intensity_percentage=0.95, seSize=None, min_pellet=24, max_pellet=12, compactness_threshold=0.75, retry_step=0.05, output_path=None):
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
                cv2.circle(image, (int(cx), int(cy)), int(diameter/2), (0, 0, 255), 2)
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

def get_avg_mm_per_pixel(detected_pellets, pellet_mm_size=6):
    sum = 0
    for (x, y, w, h), cx, cy, diameter, compactness in detected_pellets:
        sum += diameter
    avg_diameter = sum / len(detected_pellets) if detected_pellets else 0
    return pellet_mm_size / avg_diameter if avg_diameter else 0

def find_closest_pellet(detected_pellets, target_cx, target_cy):
    closest = None
    min_dist = float('inf')
    
    for (x, y, w, h), cx, cy, diameter, compactness in detected_pellets:
        if target_cx == cx and target_cy == cy:
            continue
        dist = np.sqrt((cx - target_cx) ** 2 + (cy - target_cy) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = (cx, cy, diameter)
    
    return closest

def get_pellet_ROI(image, detected_pellets, target_cx, target_cy, diameter):
    closest = find_closest_pellet(detected_pellets, target_cx, target_cy)
    if closest is None:
        return None, None, None

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    closest_cx, closest_cy, closest_diameter = closest
    closest_radius = closest_diameter / 2
    roi_radius = max(abs(target_cx-closest_cx), abs(target_cy-closest_cy)) - closest_radius

    x1 = max(0, int(target_cx - roi_radius))
    y1 = max(0, int(target_cy - roi_radius))
    x2 = min(image_width,  int(target_cx + roi_radius))
    y2 = min(image_height, int(target_cy + roi_radius))
    
    roi = image[y1:y2, x1:x2]

    # Center of the target pellet relative to the clamped sub-image
    sub_cx = target_cx - x1
    sub_cy = target_cy - y1
    
    return roi, (sub_cx, sub_cy), diameter

def main():
    file_name = '6.65.1. original.jpg'
    # file2 = '6.7.1. original.jpg'
    # file3 = '6.20.1. original.jpg'
    # file4 = '7.8.1. original.jpg'
    # file5 = 'test.jpg'
    # file6 = 'test2.jpg'
    img = read_images(f'dataset\\{file_name}')
    # img2 = read_images(f'dataset\\{file2}')
    # img3 = read_images(f'dataset\\{file3}')
    # img4 = read_images(f'dataset\\{file4}')
    # img5 = read_images(f'dataset\\{file5}')
    # img6 = read_images(f'dataset\\{file6}')

    if img is not None:
        detected = find_pellets(img, output_path=f'output\\{file_name}', seSize=7, intensity_percentage=0.9)

        rois = []
        for i, ((x, y, w, h), cx, cy, diameter, compactness) in enumerate(detected):
            roi, sub_center, diam = get_pellet_ROI(img, detected, cx, cy, diameter)
            if roi is not None:
                rois.append(roi)
                cv2.circle(img, (int(cx), int(cy)), int(diam/2), (0, 0, 255), 2)
                print(f"Pellet {i+1} sub-image center: {sub_center}")
                write_image(f'output\\{file_name}\\ROI\\pellet_{i+1}.jpg', roi)

        # size_viz = visualize_max_min_pellet(img, min_pellet=20, max_pellet=12)
        # cv2.imshow('Min / Max Pellet Size', size_viz)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found or could not be read.")
# if __name__ == "__main__":
#     main()
def get_all_centers(detected_pellets):
    """
    Extracts a list of (cx, cy) tuples from the detected_pellets list.
    """
    centers = []
    for entry in detected_pellets:
        # entry structure: ((x, y, w, h), cx, cy, diameter, compactness)
        # cx is at index 1, cy is at index 2
        cx, cy = entry[1], entry[2]
        centers.append((cx, cy))
    return centers    

# draw line and find circle edge
import cv2
import numpy as np

def count_pixels_along_line(image, img_to_draw, center, angle_degrees, max_length=100):
    # Ensure grayscale for logic
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    cx, cy = center
    angle_radians = np.radians(angle_degrees)
    end_x = int(cx + max_length * np.cos(angle_radians))
    end_y = int(cy + max_length * np.sin(angle_radians))
    
    # Generate points along the line
    line_points = np.linspace((cx, cy), (end_x, end_y), num=max_length).astype(int)
    
    found_points = []
    bg_count = 0
    last_hit_index = -100 
    
    for i, (x, y) in enumerate(line_points):
        if 0 <= y < gray_image.shape[0] and 0 <= x < gray_image.shape[1]:
            # Skip initial 20px (center of pellet)
            if i > 20:
                # Check for white pixel (>50) and cooldown (10px gap)
                if gray_image[y, x] > 50 and (i - last_hit_index >= 10):
                    cv2.circle(img_to_draw, (x, y), 2, (0, 0, 255), -1)
                    found_points.append([int(x), int(y)]) # Store as [x, y]
                    last_hit_index = i 
                    
                    if len(found_points) >= 2:
                        break

            if gray_image[y, x] <= 50:
                bg_count += 1
    
    cv2.line(img_to_draw, center, (end_x, end_y), (100, 100, 100), 1)
    
    # Return only the last point found for this specific ray
    last_point = found_points[-1] if found_points else None
    return bg_count, len(found_points), last_point

def overlay_edges(original_img, edge_map):
    # 1. Ensure edge_map is 8-bit grayscale
    if len(edge_map.shape) == 3:
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        
    # 2. Normalize if it's float64 (as your manual_canny is)
    if edge_map.dtype != np.uint8:
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # 3. Create a copy of the original to draw on
    composite = original_img.copy()
    
    # 4. Use the edge map to color pixels RED (0,0,255)
    # Wherever the edge map is white (pixel > 0), set the original image to Red
    composite[edge_map > 0] = [0, 0, 255]
    
    return composite
def calculate_average_radius_pruned(center, surface_points, threshold=1.5):
    if not surface_points: return 0.0
    cx, cy = center
    # Calculate all distances
    distances = [math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in surface_points]
    
    dist_arr = np.array(distances)
    mean_d = np.mean(dist_arr)
    std_d = np.std(dist_arr)
    
    # Prune outliers: Keep points within Mean +/- (threshold * STD)
    filtered = [d for d in distances if (mean_d - threshold*std_d) <= d <= (mean_d + threshold*std_d)]
    
    return sum(filtered) / len(filtered) if filtered else 0.0
def test():
    file_name = '6.65.1. original.jpg'
    img = read_images(f'dataset\\{file_name}')
    
    if img is None:
        return

    # 1. Processing
    img_filter_black = filter_rgb_to_black(img)[0]
    gray_frame = cv2.cvtColor(img_filter_black, cv2.COLOR_BGR2GRAY)
    canny_img = manual_canny(gray_frame, 100, 150)
    # 2. Overlay edges on the original image
    # This replaces pixels where edges are detected with Red
    composite_img = overlay_edges(img, canny_img)
    
    # 3. Pellet detection and Drawing (as you already have)
    
    
    # --- CRITICAL FIX START ---
    # Normalize the 64-bit float canny output to 8-bit (0-255)
    canny_norm = cv2.normalize(canny_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to BGR so we can draw RED (0,0,255) color on it
    canny_img_bgr = cv2.cvtColor(canny_norm, cv2.COLOR_GRAY2BGR)
    # --- CRITICAL FIX END ---

    # 2. Pellet detection
    detected = find_pellets(img, output_path=f'output\\{file_name}', seSize=7, intensity_percentage=0.9)
   
    
    # 3. Drawing
    for i, ((x, y, w, h), cx, cy, diameter, compactness) in enumerate(detected):
        # Draw on the ORIGINAL image
        cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        
        # Draw on the CANNY image
        cv2.circle(canny_img_bgr, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    # 4. Display
    cv2.imshow('Pellet Centers', img)
    cv2.imshow('Canny Output', canny_img_bgr) # Show the colorized Canny
    cv2.imshow('Overlay Result', composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test2():
    file_name = '6.65.1. original.jpg'
    img = read_images(f'dataset\\{file_name}')
    if img is None: return
    img_filter_black = filter_rgb_to_black(img)[0]
    gray_frame = cv2.cvtColor(img_filter_black, cv2.COLOR_BGR2GRAY)
    canny_img = manual_canny(gray_frame, 100, 150)
    composite_img = overlay_edges(img, canny_img)
    # ... [Your existing processing logic] ...
    canny_norm = cv2.normalize(canny_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny_img_bgr = cv2.cvtColor(canny_norm, cv2.COLOR_GRAY2BGR)

    # 2. Pellet detection
    detected = find_pellets(img, output_path=f'output\\{file_name}', seSize=7, intensity_percentage=0.9)
    
    # 3. Drawing and Analysis
    for i, pellet in enumerate(detected):
        # Unpack the structure: ((x, y, w, h), cx, cy, diameter, compactness)
        _, cx, cy, _, _ = pellet
        center_point = (int(cx), int(cy))

        # Draw center on original and canny
        cv2.circle(img, center_point, 2, (0, 0, 255), -1)
        cv2.circle(canny_img_bgr, center_point, 3, (0, 0, 255), -1)
        text = str(i)
        cv2.putText(canny_img_bgr, text, (int(cx) + 10, int(cy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        last_coor = []
        # --- NEW: Call and visualize line scan ---
        for angle in range(5, 361, 5):
            pellet_idx, _, last_pt = count_pixels_along_line(canny_img_bgr, canny_img_bgr, center_point, angle, 110)
            
            if last_pt is not None:
                last_coor.append(last_pt) # Append the [x, y] list
        # --- Analysis & Pruning ---
        # Only proceed if we have at least 36 rays hitting the surface
        if len(last_coor) >= 36:
            # Note: Changed 'outer_surface_points' to 'last_coor' to match your list name
            avg_r = calculate_average_radius_pruned(center_point, last_coor, threshold=1.5)  
            
            if avg_r > 0:
                # Draw the final result (Green circle for the average radius)
                cv2.circle(canny_img_bgr, center_point, int(avg_r), (0, 255, 0), 2)
                
                print(f"Pellet {i} Analysis:")
                print(f"  - Valid points (pre-prune): {len(last_coor)}")
                print(f"  - Average Radius: {avg_r:.2f} pixels")
            else:
                print(f"Pellet {i} rejected: Pruning removed too many outliers.")
        else:
            print(f"Pellet {i} rejected: Only {len(last_coor)} hits (Minimum 36 required).")
        # # Draw the line on the image to see what we are checking
        # dx = np.cos(np.radians(angle))
        # dy = np.sin(np.radians(angle))
        # end_point = (int(cx + 100 * dx), int(cy + 100 * dy))
        # cv2.line(canny_img_bgr, center_point, end_point, (255, 255, 0), 1)

    # 4. Display
    cv2.imshow('Pellet Centers', img)
    cv2.imshow('Canny Output', canny_img_bgr)
    cv2.imshow('Overlay Result', composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
test2()

