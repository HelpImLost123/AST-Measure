import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

def filter_rgb_to_black(image):
    # Check if the image is color (3 channels) or grayscale (2D array/1 channel)
    lower = 90
    if len(image.shape) == 3:
        # It's a color image (BGR)
        lower_range = np.array([lower, lower, lower])
        upper_range = np.array([255, 255, 255])
    else:
        # It's a grayscale image
        lower_range = np.array([lower])
        upper_range = np.array([255])

    mask = cv2.inRange(image, lower_range, upper_range)
    
    result = image.copy()
    
    # If color, use [0, 0, 0], if grayscale, use 0
    black_color = [0, 0, 0] if len(image.shape) == 3 else 0
    result[mask > 0] = black_color
    

    return result, mask
def manual_canny(img_gray, low_threshold=50, high_threshold=150):
    # 1. Invert the image (since you wanted it inverted)
    
    img_invert = cv2.bitwise_not(img_gray)
    # 2. Noise Reduction
    mask = cv2.inRange(img_invert, 156, 175)
    img_blur = cv2.GaussianBlur(mask, (9, 9), 10)
    median_result = cv2.medianBlur(img_blur, 13)
    mask2 = cv2.inRange(median_result, 0, 10)
    

    # 3. Gradient Calculation
    gx = cv2.Sobel(np.float32(mask2), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(mask2), cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # FIX: Get shape from img_blur instead of the deleted 'mask'
    height, width = mask2.shape
    nms = np.zeros_like(mag)
    
    # 4. Non-Maximum Suppression
    ang = ang % 180 
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q, r = 255, 255
            if (0 <= ang[i,j] < 22.5) or (157.5 <= ang[i,j] <= 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif (22.5 <= ang[i,j] < 67.5):
                q, r = mag[i+1, j-1], mag[i-1, j+1]
            elif (67.5 <= ang[i,j] < 112.5):
                q, r = mag[i+1, j], mag[i-1, j]
            elif (112.5 <= ang[i,j] < 157.5):
                q, r = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i,j] >= q and mag[i,j] >= r:
                nms[i,j] = mag[i,j]
            else:
                nms[i,j] = 0

    # 5. Double Thresholding
    res = np.zeros_like(nms)
    # mask edge in white
    res[nms >= high_threshold] = 255
    res[(nms >= low_threshold) & (nms < high_threshold)] = 50 
    
    return res


def process_dataset(folder_path):
    # 1. Get all .jpg files from the folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not file_list:
        print(f"No images found in {folder_path}")
        return

    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        frame1 = cv2.imread(img_path)
        gray_for_otsu = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_for_otsu, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame = filter_rgb_to_black(frame1)[0]
        
        if frame is None:
            print(f"Skipping {filename}: Could not read image.")
            continue

        # --- Your Processing Logic ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny (ensure your manual_canny returns uint8 as discussed before!)
        canny_img = manual_canny(gray_frame, 100, 150)
        
        # Image Inversion and Masking
        img_invert = cv2.bitwise_not(gray_frame)
        mask = cv2.inRange(img_invert, 156, 175)
        
        # Noise Reduction
        img_blur = cv2.GaussianBlur(mask, (9, 9), 10)
        median_result = cv2.medianBlur(img_blur, 13)
        mask2 = cv2.inRange(median_result, 0, 10)

        # --- Visualization ---
        plt.figure(figsize=(15, 4))
        plt.suptitle(f"Processing: {filename}")
        
        plt.subplot(1, 5, 1)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.title('rgb_to_black (90-255 to black)')
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 5, 3)    
        plt.title('Canny Edges')
        plt.imshow(canny_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 4)    
        plt.title('Inverted Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 5)    
        plt.title('Final Median')
        plt.imshow(mask2, cmap='gray')
        plt.axis('off')

        plt.show()


def process_dataset2(folder_path):
    # 1. Get all .jpg files from the folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not file_list:
        print(f"No images found in {folder_path}")
        return

    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        #original image
        frame1 = cv2.imread(img_path)
        # 1 gray for otsu
        gray_for_otsu = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_for_otsu, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2 thresholding
        frame = filter_rgb_to_black(frame1)[0]
        
        if frame is None:
            print(f"Skipping {filename}: Could not read image.")
            continue
        
        # --- Your Processing Logic ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny (ensure your manual_canny returns uint8 as discussed before!)
        canny_img = manual_canny(gray_frame, 100, 150)
        
        # 3 Image Inversion and Masking
        img_invert = cv2.bitwise_not(gray_frame)
        mask = cv2.inRange(img_invert, 156, 175)
        # 4 Cleaning the Mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 2. Clean up small white noise dots
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3. Use mask_cleaned for your contours instead of the raw mask
        contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Noise Reduction
        img_blur = cv2.GaussianBlur(mask, (9, 9), 10)
        median_result = cv2.medianBlur(img_blur, 13)
        mask2 = cv2.inRange(median_result, 0, 10)

        # --- Visualization ---
        plt.figure(figsize=(18, 4)) # Increased width for the extra plot
        plt.suptitle(f"Processing: {filename}")
        
        # 1. Original
        plt.subplot(1, 6, 1)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 2. Processed
        plt.subplot(1, 6, 2)
        plt.title('rgb_to_black')
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 3. Canny
        plt.subplot(1, 6, 3)    
        plt.title('Canny Edges')
        plt.imshow(canny_img, cmap='gray')
        plt.axis('off')

        # 4. Inverted Mask
        plt.subplot(1, 6, 4)    
        plt.title('Inverted Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        # 5. Final Median
        plt.subplot(1, 6, 5)    
        plt.title('Final Median')
        plt.imshow(mask2, cmap='gray')
        plt.axis('off')

        # 6. Contour Overlay (NEW)
        plt.subplot(1, 6, 6)
        canny_8bit = cv2.convertScaleAbs(canny_img) 
        # 2. Convert to BGR so we can draw colors on it
        canny_bgr = cv2.cvtColor(canny_8bit, cv2.COLOR_GRAY2BGR)
        contour_overlay = canny_bgr.copy()
        # Draw all contours in green
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
        # Draw largest contour in red
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_overlay, [cnt], -1, (0, 0, 255), 3)
            
        plt.title('Detected Contours')
        plt.imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()
# Run the function
def process_main():
    # --- Main Execution ---
    frame = cv2.imread('dataset/6.65.1. original.jpg')
    frame = filter_rgb_to_black(frame)[0]

    # FIX: Check if frame exists BEFORE doing any image processing!
    if frame is None:
        print("Error: image not found! Please check if 'test.jpg' is in the correct folder.")
    else:
        # Convert to grayscale first
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Pass the grayscale frame into our fixed function
        canny_img = manual_canny(gray_frame, 100, 150)
        
        
        # Create the inverted image for the plot display
        img_invert = cv2.bitwise_not(gray_frame)
        # mask = cv2.inRange(img_invert, 109, 136)
        mask = cv2.inRange(img_invert, 156, 175)
        
        # median_result = cv2.medianBlur(mask, 13)
        img_blur = cv2.GaussianBlur(mask, (9, 9), 10)
        median_result = cv2.medianBlur(img_blur, 13)
        mask2 = cv2.inRange(median_result, 30, 255)
        
        # mask = cv2.inRange(img_invert, np.array([109, 109, 109]), np.array([136, 136, 136]))

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title('Input Image')
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 4, 2)    
        plt.title('Canny Edges')
        plt.imshow(canny_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)    
        plt.title('Inverted Grayscale')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)    
        plt.title('median')
        plt.imshow(mask2, cmap='gray')
        plt.axis('off')

        plt.show()
        
# process_dataset('dataset')
# process_dataset2('dataset')
# process_main()