import pytest
import numpy as np
from detector_neumonia import read_dicom_file

@pytest.fixture
def ejemplo_dicom_path():
    # Define una ruta de ejemplo a un archivo DICOM en tu sistema
    return 'ruta/a/tu/archivo.dcm'

def test_read_dicom_file(ejemplo_dicom_path):
    # Ejecuta la función read_dicom_file con la ruta de ejemplo
    img_RGB, img2show = read_dicom_file(ejemplo_dicom_path)
    
    # Verifica que img_RGB y img2show no sean None
    assert img_RGB is not None, "La imagen RGB resultante no debería ser None"
    assert img2show is not None, "La imagen img2show no debería ser None"

    # Verifica el tipo de img_RGB y su forma
    assert isinstance(img_RGB, np.ndarray), "img_RGB debería ser un ndarray de NumPy"
    assert img_RGB.shape[-1] == 3, "img_RGB debería tener 3 canales de color (RGB)"

    # Verifica que img2show sea una instancia de PIL.Image.Image
    from PIL import Image
    assert isinstance(img2show, Image.Image), "img2show debería ser una instancia de PIL.Image.Image"

    # Puedes agregar más aserciones según las características específicas que desees verificar