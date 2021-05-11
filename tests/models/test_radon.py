import numpy as np
import pytest
from helpers import mod_image, ref_image

from imgreg.models.radon import RadonParams, RadonSolver


@pytest.fixture
def radon_solver(ref_image, mod_image):
    ras = RadonSolver(ref_image, mod_image)
    return ras


class TestRadonSolver:
    @staticmethod
    @pytest.mark.parametrize(
        "img_params, value, rtol",
        (
            [RadonParams.ANGULAR_PRECISION, 0.2, 1e-16],
            [RadonParams.EXPONENTIAL_FILTER_SIGNAL_NOISE, 100, 1e-16],
            [RadonParams.UPSAMPLING, 10, 1e-16],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_param(radon_solver, img_params, value, rtol):
        assert radon_solver[img_params].value == value

    @staticmethod
    @pytest.mark.parametrize(
        "img_params, value, rtol",
        (
            [RadonParams.THETA, np.arange(0, 180, 1), 1e-16],
            [RadonParams.REF_ROTATION, np.array([179.67901235, 0.2]), 1e-16],
            [RadonParams.MOD_ROTATION, np.array([168.32098765, 0.2]), 1e-16],
            [
                RadonParams.ROTATION_CANDIDATE,
                np.array([-11.35802469, 0.28284271]),
                1e-16,
            ],
            [
                RadonParams.RECOVERED_ROTATION,
                np.array([-11.35802469, 0.28284271]),
                1e-16,
            ],
            [
                RadonParams.RECOVERED_TRANSLATION,
                np.array([-11.80822224, 25.77936941, 0.30466765]),
                1e-16,
            ],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_np_param(radon_solver, img_params, value, rtol):
        assert not (
            False in np.isclose(radon_solver[img_params].value, value, rtol=rtol)
        )

    @staticmethod
    @pytest.mark.parametrize(
        "img_params",
        (
            [RadonParams.REF_IMG, RadonParams.MOD_IMG],
            [RadonParams.RECOVERED_ROT_IMG, RadonParams.REF_IMG],
            [RadonParams.RECOVERED_ROT_TR_IMG, RadonParams.REF_IMG],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_display(radon_solver, img_params):
        radon_solver.display(img_params)
