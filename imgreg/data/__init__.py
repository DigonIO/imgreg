"""Test images."""
import pathlib

from PIL import Image

path = pathlib.Path(__file__).parent.absolute()


def ref_img() -> Image:
    """Provides a reference image."""
    return Image.open(f"{path}/ref_img.jpg")


def mod_img() -> Image:
    """Provides the modified reference image variant."""
    return Image.open(f"{path}/mod_img.jpg")
