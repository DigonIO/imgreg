from typing import Any, Set

import math as m
import numpy as np
import pytest
from helpers import retina

from imgreg.util.methods import ImageMethods


@pytest.mark.parametrize(
    "angle_scale_translation, expected_err, rtol",
    (
        [
            [(0, 1, (0, 0)), 1.6042220747667237e-13, 1e-02],
            [(0, 1, (31, 20)), 1.6042220747667237e-13, 1e-02],
            [(360, 1, (31, 20)), 4.0453985364827997e-13, 1e-02],
            [(20, 1, (0, 0)), 0.466197654596354, 1e-02],
            [(50, 1.2, (0, 0)), 1.0588670761092247, 1e-02],
            [(70, 2, (0, 0)), 3.8213151188382875, 1e-02],
            [(12, 1, (70, 90)), 0.4629894180729796, 1e-02],
            [(33.1, 0.8, (4.5, 10)), 0.44925638035010895, 1e-04],
            [(190, 1.1, (1.1, 20)), 0.7235503626182368, 1e-02],
        ]
    ),
)
class TestImageMethods:
    def test_compute_rts(self, retina, angle_scale_translation, expected_err, rtol):
        angle, scale, translation = angle_scale_translation
        mod_img = ImageMethods.compute_rts(
            retina, angle=angle, scale=scale, translation=translation
        )
        rec_img = ImageMethods.compute_rts(
            mod_img, angle=angle, scale=scale, translation=translation, inverse=True
        )
        assert m.isclose(np.linalg.norm(retina - rec_img), expected_err, rel_tol=rtol)
