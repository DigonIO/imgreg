"""Test images."""
from PIL import Image
import pathlib

path = pathlib.Path(__file__).parent.absolute()


def ref_img():
    return Image.open(f"{path}/ref_img.jpg")


def mod_img():
    return Image.open(f"{path}/mod_img.jpg")
