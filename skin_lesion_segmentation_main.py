import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to extract lesions from a given image
def extract_lesions(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(10, 10))
    contrasted = clahe.apply(grayscale_image)

    # Smooth with Gaussian blur
    smoothed = cv2.GaussianBlur(contrasted, (9, 9), 0)

    # Apply thresholding
    _, binary_image = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Smooth the segmented lesion
    smoothed_lesion = cv2.dilate(closed, kernel, iterations=1)
    smoothed_lesion = cv2.erode(smoothed_lesion, kernel, iterations=1)

    # Remove small contours
    contours, _ = cv2.findContours(smoothed_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Adjust as needed
            cv2.drawContours(smoothed_lesion, [contour], 0, 0, -1)

    return image, smoothed_lesion

# Main function
def main():
    folder_path = 'melanoma'  # Folder containing input images
    images_data = []  # List to store tuples of (original_image, segmented_image)

    # Process all images in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Only process supported image files
        if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
            continue

        try:
            # Extract lesions from the image
            original_image, lesion_image = extract_lesions(image_path)
            images_data.append((original_image, lesion_image))
        except ValueError as e:
            print(e)

    # Display all images in a grid
    num_images = len(images_data)
    if num_images == 0:
        print("No valid images found in the folder.")
        return

    fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))
    if num_images == 1:  # Special case for a single image
        axes = [axes]

    for i, (original_image, lesion_image) in enumerate(images_data):
        axes[i][0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"Original Image {i+1}")
        axes[i][0].axis('off')

        axes[i][1].imshow(lesion_image, cmap='gray')
        axes[i][1].set_title(f"Extracted Lesion {i+1}")
        axes[i][1].axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
