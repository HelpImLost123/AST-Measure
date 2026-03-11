import numpy as np
import cv2
import matplotlib.pyplot as plt

def manual_canny(img_color, low_threshold=50, high_threshold=150):
    # 1. Color Filtering (Use the BGR color image here)
    # Note: If the image is inverted, your target values might change!
    img_invert = cv2.bitwise_not(img_color)
    
    lower_gray = np.array([156, 156, 156])
    upper_gray = np.array([175, 175, 175])
    
    # This creates a binary mask (Black and White)
    mask = cv2.inRange(img_invert, lower_gray, upper_gray)
    
    
    # 2. Noise Reduction
    img_blur = cv2.GaussianBlur(mask, (9, 9), 10)

    
    # 3. Gradient Calculation
    gx = cv2.Sobel(np.float32(img_blur), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(img_blur), cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Use the mask shape (which is 2D: height, width)
    height, width = mask.shape
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
    res[nms >= high_threshold] = 255
    res[(nms >= low_threshold) & (nms < high_threshold)] = 50 
    
    return res

# --- Main Execution ---
frame = cv2.imread('6.65.1. original.jpg')

if frame is None:
    print("Error: image not found!")
else:
    # Pass the COLOR frame, not the grayscale one
    canny_img = manual_canny(frame, 100, 150)
    
    # Create the mask separately just for the plot display
    img_invert = cv2.bitwise_not(frame)
    mask = cv2.inRange(img_invert, np.array([156,156,156]), np.array([175,175,175]))
    img_blur = cv2.GaussianBlur(mask, (9, 9), 10)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)    
    plt.title('Canny Edges (from Mask)')
    plt.imshow(canny_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)    
    plt.title('Color Range Mask(Thresholding)')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()
    
    
# canny_img = manual_canny(frame, 50, 150)
# canny_img_bin = np.uint8(canny_img)
# lines = cv2.HoughLinesP(canny_img_bin, 1, np.pi/180, 68, minLineLength=15, maxLineGap=250)

# for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
# # Show result
# print("Line Detection using Hough Transform")
# cv2.imshow('lanes',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()