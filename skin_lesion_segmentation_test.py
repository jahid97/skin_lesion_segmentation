import unittest
import os
import cv2
from unittest.mock import patch
import numpy as np
from skin_lesion_segmentation_main import extract_lesions, main  # Ensure correct import

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        """Set up the test environment, creating necessary files."""
        self.test_image_path_valid = 'melanoma/ISIC_0024306.jpg'
        self.test_image_path_invalid = 'invalid_image.png'
        
        # Create a dummy image for testing (a black square image)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Save the dummy image
        cv2.imwrite(self.test_image_path_valid, self.test_image)
        
        # Check if the image was saved correctly
        if cv2.imread(self.test_image_path_valid) is None:
            raise ValueError(f"Failed to save or read the valid test image at {self.test_image_path_valid}")

    def tearDown(self):
        """Clean up after tests by removing the test image files."""
        if os.path.exists(self.test_image_path_valid):
            os.remove(self.test_image_path_valid)

    def test_extract_lesions_valid(self):
        """Test that extract_lesions processes a valid image correctly."""
        original_image, lesion_image = extract_lesions(self.test_image_path_valid)
        
        # Check that the extracted images are not None
        self.assertIsNotNone(original_image)
        self.assertIsNotNone(lesion_image)
        
        # Check if lesion_image is a binary image (contains only 0 or 255)
        unique_values = np.unique(lesion_image)
        self.assertTrue(np.all(np.isin(unique_values, [0, 255])))

    def test_extract_lesions_invalid_image(self):
        """Test that extract_lesions raises a ValueError for invalid image."""
        with self.assertRaises(ValueError):
            extract_lesions(self.test_image_path_invalid)

    @patch('os.listdir')
    @patch('cv2.imread')
    def test_main_function(self, mock_imread, mock_listdir):
        """Test the main function to check if it handles multiple images."""
        # Mock the list of files and imread behavior
        mock_listdir.return_value = ['test_image.png']
        mock_imread.return_value = self.test_image

        # Call the main function
        try:
            main()  # This will run the main function and check if it raises any exceptions
        except Exception as e:
            self.fail(f"Main function raised an exception: {e}")

    @patch('cv2.imread')
    def test_invalid_image_format(self, mock_imread):
        """Test handling of unsupported image formats."""
        mock_imread.return_value = None  # Simulate failure in reading the image
        
        unsupported_image_path = 'test_image.bmp'
        with self.assertRaises(ValueError):
            extract_lesions(unsupported_image_path)

if __name__ == '__main__':
    unittest.main()
