import pytest
import cv2
from PIL import Image

from detector_neumonia import read_jpg_file

def test_read_jpg_file():
    # Ruta de acceso al archivo
    jpg_file_path = 'ruta/a/tu/imagen.jpg'

    # Llamado a la función de prueba
    img2, img2show = read_jpg_file(jpg_file_path)

    # Verificación de salida
    assert isinstance(img2, np.ndarray), "img2 should be a numpy array"
    assert isinstance(img2show, Image.Image), "img2show should be an Image object"
    assert img2.shape[2] == 3, "img2 should have 3 channels (RGB)"

if __name__ == "__main__":
    pytest.main()