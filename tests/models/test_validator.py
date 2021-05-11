import math as m
import pytest
from helpers import ref_image

from imgreg.models.validator import Validator, ValidatorParams
from imgreg.util.methods import ImageMethods


@pytest.fixture
def request_rts(request, ref_image):
    angle, translation = request.param
    return ImageMethods.compute_rts(ref_image, angle=angle, translation=translation)


class TestValidator:
    @staticmethod
    @pytest.mark.parametrize(
        "request_rts",
        (
            (2, (6, 2)),
            (1, (1, 2)),
        ),
        indirect=["request_rts"],
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_display(tmpdir_factory, request_rts, ref_image):
        val = Validator(request_rts, ref_image)
        val.display(
            [
                ValidatorParams.IMG,
                ValidatorParams.REF_IMG,
            ]
        )
        val.display(
            [
                ValidatorParams.ABSOLUTE_DIFFERENCE_IMG,
                ValidatorParams.SQUARED_DIFFERENCE_IMG,
            ]
        )

    @staticmethod
    @pytest.mark.parametrize(
        "request_rts, norm_rel_l2, rtol",
        (
            [(2, (6, 2)), 0.4875079942792268, 1e-13],
            [(1, (1, 2)), 0.3942652180108632, 1e-13],
        ),
        indirect=["request_rts"],
    )
    def test_norm_rel_l2(request_rts, ref_image, norm_rel_l2, rtol):
        val = Validator(request_rts, ref_image)
        assert m.isclose(
            val[ValidatorParams.NORM_REL_L2].value, norm_rel_l2, rel_tol=rtol
        )
