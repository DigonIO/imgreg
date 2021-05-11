import numpy as np
import pytest
from helpers import mod_image, ref_image, validator

from imgreg.models.logpolar import LogPolarSolver, LogPolParams
from imgreg.models.logpolar.params import (
    BoundsParam,
    FourierModParam,
    FourierRefParam,
    GaussDiffParam,
    GaussDiffWindowModParam,
    GaussDiffWindowRefParam,
    ModImageParam,
    RecoveredRotationParam,
    RecoveredRotationScalePhaseParam,
    RecoveredRotScaleParam,
    RecoveredRotScaleTr,
    RecoveredScaleParam,
    RecoveredTranslationParam,
    RefImageParam,
    UpsamplingParam,
    WarpedFourierModParam,
    WarpedFourierRefParam,
    WindowRadiusExpParam,
    WindowTypeParam,
    WindowWeightParam,
)
from imgreg.models.validator import ValidatorParams


@pytest.fixture
def log_polar_solver(ref_image, mod_image):
    irs = LogPolarSolver(ref_image, mod_image)
    return irs


class TestImRegSolver:
    @staticmethod
    def test_l2_validation(log_polar_solver, validator):
        validator[ValidatorParams.IMG].value = log_polar_solver[
            LogPolParams.RECOVERED_ROT_SCALE_TR_IMG
        ].value
        assert validator[ValidatorParams.NORM_REL_L2].value < 0.5891

    @staticmethod
    def test_tuned_l2_validation(log_polar_solver, validator):
        log_polar_solver[LogPolParams.GAUSS_DIFF].value = (5, 31)
        log_polar_solver[LogPolParams.UPSAMPLING].value = 20
        validator[ValidatorParams.IMG].value = log_polar_solver[
            LogPolParams.RECOVERED_ROT_SCALE_TR_IMG
        ].value
        assert validator[ValidatorParams.NORM_REL_L2].value < 0.4682

    @staticmethod
    @pytest.mark.parametrize(
        "img_params, value",
        (
            [LogPolParams.GAUSS_DIFF, (5, 20)],
            [LogPolParams.WINDOW_WEIGHT, 1],
            [LogPolParams.WINDOW_TYPE, "hann"],
            [LogPolParams.WINDOW_RADIUS_EXP, 3],
            [
                LogPolParams.BOUNDS,
                (341, 491, 225, 375),
            ],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_param(log_polar_solver, img_params, value):
        assert log_polar_solver[img_params].value == value

    @staticmethod
    @pytest.mark.parametrize(
        "img_params, value, rtol",
        (
            [
                LogPolParams.RECOVERED_ROTATION,
                np.array([-13.06730769, 0.11259774]),
                1e-16,
            ],
            [LogPolParams.RECOVERED_SCALE, np.array([1.0, 1.00187429]), 1e-16],
            [
                LogPolParams.RECOVERED_TRANSLATION,
                np.array([-17.98318062, 31.037803, 0.42407651]),
                1e-16,
            ],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_np_param(log_polar_solver, img_params, value, rtol):
        assert not (
            False in np.isclose(log_polar_solver[img_params].value, value, rtol=rtol)
        )

    @staticmethod
    @pytest.mark.parametrize(
        "img_params",
        (
            [LogPolParams.REF_IMG, LogPolParams.MOD_IMG],
            [LogPolParams.GAUSS_DIFF_REF_IMG, LogPolParams.GAUSS_DIFF_MOD_IMG],
            [LogPolParams.FOURIER_REF_IMG, LogPolParams.FOURIER_MOD_IMG],
            [LogPolParams.WARPED_FOURIER_REF_IMG, LogPolParams.WARPED_FOURIER_MOD_IMG],
            [LogPolParams.RECOVERED_ROT_SCALE_IMG, LogPolParams.REF_IMG],
            [LogPolParams.RECOVERED_ROT_SCALE_TR_IMG, LogPolParams.REF_IMG],
        ),
    )
    @pytest.mark.filterwarnings("ignore:Matplotlib")
    def test_display(log_polar_solver, img_params):
        log_polar_solver.display(img_params)


class TestParams:
    @staticmethod
    def test_image_param(ref_image):
        p_ref = RefImageParam(ref_image)
        p_ref.display()

    class TestBoundsParam:
        @staticmethod
        def test_init():
            p_bounds = BoundsParam({}, bounds=(1, 2, 1, 2))

        @staticmethod
        def test_value_function_handle():
            pass
