import numpy as np
import pytest
from skimage import data as ski_data
import imgreg.data as data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib

from imgreg.models.validator import Validator

matplotlib.use("AGG")


@pytest.fixture
def validator(ref_image):
    return Validator(ref_image, ref_image)


@pytest.fixture
def retina():
    retina = rgb2gray(ski_data.retina())
    retina_small = resize(retina, np.array(retina.shape) // 2)
    return np.pad(retina_small, 128, "constant", constant_values=0)


@pytest.fixture
def ref_image():
    return np.array(data.ref_img())


@pytest.fixture
def mod_image():
    return np.array(data.mod_img())
