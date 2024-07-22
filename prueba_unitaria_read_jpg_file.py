import pytest
import cv2
from PIL import Image

from detector_neumonia import read_jpg_file

# Test case 1: Test read_jpg_file with a valid JPG file
def test_read_jpg_file():
    # Replace with the path to a JPG file in your test environment
    jpg_file_path = 'ruta/a/tu/imagen.jpg'

    # Call the function under test
    img2, img2show = read_jpg_file(jpg_file_path)

    # Assertions to verify the function output
    assert isinstance(img2, np.ndarray), "img2 should be a numpy array"
    assert isinstance(img2show, Image.Image), "img2show should be an Image object"
    assert img2.shape[2] == 3, "img2 should have 3 channels (RGB)"

if __name__ == "__main__":
    pytest.main()