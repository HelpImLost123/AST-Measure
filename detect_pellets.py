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

def binarize_image(image, threshold=230):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to the grayscale image
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image

def erode_image(image, kernel_size=3):
    # Create a structuring element (kernel) for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply erosion to the image using the kernel
    eroded_image = cv2.erode(image, kernel, iterations=1)
    
    return eroded_image

def dilate_image(image, kernel_size=3):
    # Create a structuring element (kernel) for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply dilation to the image using the kernel
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    return dilated_image

def close_image(image, kernel_size=3):
    # Create a structuring element (kernel) for closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply closing to the image using the kernel
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image

def open_image(image, kernel_size=3):
    # Create a structuring element (kernel) for opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply opening to the image using the kernel
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return opened_image

def find_pellets(image, intensity_percentage=0.95, min_pallet=30, max_pallet=4, compactness_threshold=0.75, retry_step=0.05, output_path=None):
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
    

    minPelletDiameter = int((min(gray.shape) / min_pallet))
    maxPelletDiameter = int((min(gray.shape) / max_pallet))
    
    seSize = max(3, int(minPelletDiameter / 10))
    print(f"Structuring Element Size: {seSize}")

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seSize, seSize))

    img_h, img_w = image.shape[:2]
    threshold = intensity_percentage
    
    detected = []

    while threshold >= 0:
        print(f"Trying intensity threshold: {threshold:.2f}")

        _, binary = cv2.threshold(equalized, 255 * threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Binary Image', binary)
        
        opened_image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        # cv2.imshow('Opened Image', opened_image)
        
        if output_path:
            write_image(f'{output_path}\\1_gray.jpg', gray)
            write_image(f'{output_path}\\2_equalized.jpg', equalized)
            write_image(f'{output_path}\\3_binary.jpg', binary)
            write_image(f'{output_path}\\4_opened.jpg', opened_image)
        
        contours = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in contours:
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, 1, True)

            (cx, cy), radius = cv2.minEnclosingCircle(approx)
            diameter = radius * 2

            if cx - radius < 0 or cy - radius < 0 or cx + radius > img_w or cy + radius > img_h:
                continue
            if diameter < minPelletDiameter or diameter > maxPelletDiameter:
                continue

            enclosing_area = np.pi * radius ** 2
            compactness = cv2.contourArea(approx) / enclosing_area if enclosing_area > 0 else 0
            if compactness < compactness_threshold:
                continue

            detected.append((cv2.boundingRect(approx), cx, cy, diameter, compactness))

        if detected:
            for (x, y, w, h), cx, cy, diameter, compactness in detected:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(f"Detected pellet | cx: {cx}, cy: {cy}, diameter: {diameter:.1f} px, compactness: {compactness:.2f}")
            if output_path:
                write_image(f'{output_path}\\5_detected.jpg', image)
            break

        print(f"No pellets found at threshold {threshold:.2f}, retrying...")
        threshold -= retry_step
    else:
        print("No pellets detected after exhausting all thresholds.")

    cv2.imshow('Detected Pellets', image)
    
    return detected
    
    

def main():
    file_name = '6.63.1. original.jpg'
    img = read_images(f'dataset\\{file_name}')

    if img is not None:
        detected = find_pellets(img, output_path=f'output\\{file_name}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found or could not be read.")
        
if __name__ == "__main__":
    main()