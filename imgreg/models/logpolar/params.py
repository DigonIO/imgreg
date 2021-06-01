"""
Module implementing the parameter classes for the LogPolarSolver Model

Author: Fabian A. Preiss
"""

from enum import Enum
from typing import Dict, Hashable, Optional, Tuple, cast, Type, Any

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform

from imgreg.models.logpolar.enums import LogPolParams
from imgreg.util.methods import ImageMethods
from imgreg.util.params import ImageParameter, Parameter

# Root Parameters


class RefImageParam(ImageParameter):
    """
    Reference image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        super().__init__(LogPolParams.REF_IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Reference Image"


class ModImageParam(ImageParameter):
    """
    Modified image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        super().__init__(LogPolParams.MOD_IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Modified Image"


class GaussDiffParam(Parameter):
    """
    Lower and upper kernel size of the DoG filter.

    .. _wgaussdiff:
        https://en.wikipedia.org/wiki/Difference_of_Gaussians

    Attributes
    ----------
    value : tuple[float, float]

    Notes
    -----
    Typical ratios for image enhancement are in the order of 1:4 and 1:5. [#f3]_
    A ratio of 1.6 approximates the Laplacian of Gaussian filter.

    References
    ----------
    .. [#f3] `Wikipedia, "Difference of Gaussians" <wgaussdiff_>`_
    """

    def __init__(self, gauss_diff: Tuple[float, float]):
        # NOTE mypy throws: Argument 2 to "__init__" of "Parameter" has incompatible type "object";
        # therefore weird casting of Tuple[float, float] to Type[Any]
        super().__init__(LogPolParams.GAUSS_DIFF, cast(Type[Any], Tuple[float, float]))
        self.value = gauss_diff


class WindowWeightParam(Parameter):
    """
    Weighting factor scaling beween windowed image and image, range of [0,1].

    Attributes
    ----------
    value : str
    """

    def __init__(self, window_weight: float):
        super().__init__(LogPolParams.WINDOW_WEIGHT, float)
        self.value = window_weight


class WindowTypeParam(Parameter):
    """
    Window type for FFT filter, see
    `scipy.signal.windows.get_window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html>`_
    for possible choices.

    Attributes
    ----------
    value : str
    """

    def __init__(self, window_type: str):
        super().__init__(LogPolParams.WINDOW_TYPE, str)
        self.value = window_type


class WindowRadiusExpParam(Parameter):
    """
    Window radius exponent, larger wrexp => faster computation.

    Notes
    -----
    If a value larger then 1 is used, this introduces a lowpass filter.

    Attributes
    ----------
    value : float >= 1
    """

    def __init__(self, w_r_exp: float):
        super().__init__(LogPolParams.WINDOW_RADIUS_EXP, float)
        self.value = w_r_exp


class UpsamplingParam(Parameter):
    """
    Upsampling factor. 1 => no upsampling, 20 => precision to 1/20 of a pixel.

    Attributes
    ----------
    value : int
    """

    def __init__(self, upsampling: int):
        super().__init__(LogPolParams.UPSAMPLING, int)
        self.value = upsampling


# Derived Parameters


# NOTE: design currently leads to a full clearing of derived properties although
# bounds is only an optical property and not necessary for the algorithm
class BoundsParam(Parameter):
    """
    Boundaries for image slicing.

    Attributes
    ----------
    value : tuple[int, int, int, int]
    """

    def __init__(
        self, parent_parameters, bounds: Optional[Tuple[int, int, int, int]] = None
    ):
        super().__init__(
            LogPolParams.BOUNDS,
            BoundsParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if bounds is not None:
            self.value = bounds

    @staticmethod
    def _value_function_handle(
        params: Dict[Enum, Parameter]
    ) -> Tuple[int, int, int, int]:
        image = params[LogPolParams.REF_IMG].value
        wrexp = params[LogPolParams.WINDOW_RADIUS_EXP].value
        warp_radius = ImageMethods.compute_warp_radius(min(image.shape), wrexp)
        center = np.array(image.shape) // 2
        r_lower, r_upper = (
            center[0] - warp_radius,
            center[0] + warp_radius,
        )
        c_lower, c_upper = (
            center[1] - warp_radius,
            center[1] + warp_radius,
        )
        return (r_lower, r_upper, c_lower, c_upper)


def _calculate_gaussdiff(image_type: Enum, params: Dict[Enum, Parameter]) -> np.ndarray:
    return ImageMethods.compute_dgfw(
        params[image_type].value,
        params[LogPolParams.GAUSS_DIFF].value,
        params[LogPolParams.WINDOW_WEIGHT].value,
        params[LogPolParams.WINDOW_TYPE].value,
    )


class GaussDiffWindowRefParam(ImageParameter):
    """
    Gaussian difference filtered and windowed reference input image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, gauss_diff_ref_img=None):
        super().__init__(
            LogPolParams.GAUSS_DIFF_REF_IMG,
            GaussDiffWindowRefParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Filtered Reference Image"
        if gauss_diff_ref_img is not None:
            self.value = gauss_diff_ref_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return _calculate_gaussdiff(LogPolParams.REF_IMG, params)


class GaussDiffWindowModParam(ImageParameter):
    """
    Gaussian difference filtered and windowed modified input image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, gauss_diff_mod_img=None):
        super().__init__(
            LogPolParams.GAUSS_DIFF_MOD_IMG,
            GaussDiffWindowModParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Filtered Modified Image"
        if gauss_diff_mod_img is not None:
            self.value = gauss_diff_mod_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return _calculate_gaussdiff(LogPolParams.MOD_IMG, params)


class FourierRefParam(ImageParameter):
    """
    Fourier transformed and filtered reference image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, fourier_ref_img=None):
        super().__init__(
            LogPolParams.FOURIER_REF_IMG,
            FourierRefParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Reference FFT Image\n(magnitude; zoomed)"
        self.cmap = "magma"
        if fourier_ref_img is not None:
            self.value = fourier_ref_img
        self.set_bounds_lookup(LogPolParams.BOUNDS)

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_afts(params[LogPolParams.GAUSS_DIFF_REF_IMG].value)


class FourierModParam(ImageParameter):
    """
    Fourier transformed and filtered modified image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, fourier_mod_img=None):
        super().__init__(
            LogPolParams.FOURIER_MOD_IMG,
            FourierModParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Modified FFT Image\n(magnitude; zoomed)"
        self.cmap = "magma"
        if fourier_mod_img is not None:
            self.value = fourier_mod_img
        self.set_bounds_lookup(LogPolParams.BOUNDS)

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_afts(params[LogPolParams.GAUSS_DIFF_MOD_IMG].value)


class WarpedFourierRefParam(ImageParameter):
    """
    Log-polar transformed of the fourier transformed reference image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, warped_fourier_ref_img=None):
        super().__init__(
            LogPolParams.WARPED_FOURIER_REF_IMG,
            WarpedFourierRefParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Reference FFT Image\n(Log-Polar-Transformed)"
        self.cmap = "magma"
        if warped_fourier_ref_img is not None:
            self.value = warped_fourier_ref_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_log_polar_tf(
            params[LogPolParams.FOURIER_REF_IMG].value,
            params[LogPolParams.WINDOW_RADIUS_EXP].value,
        )


class WarpedFourierModParam(ImageParameter):
    """
    Log-polar transformed of the fourier transformed modified image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, warped_fourier_mod_img=None):
        super().__init__(
            LogPolParams.WARPED_FOURIER_MOD_IMG,
            WarpedFourierModParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Modified FFT Image\n(Log-Polar-Transformed)"
        self.cmap = "magma"
        if warped_fourier_mod_img is not None:
            self.value = warped_fourier_mod_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_log_polar_tf(
            params[LogPolParams.FOURIER_MOD_IMG].value,
            params[LogPolParams.WINDOW_RADIUS_EXP].value,
        )


class RecoveredRotationScalePhaseParam(Parameter):
    """
    Recovered rotation angle in degrees and scaling factor including errors.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much
    larger depending on the data.

    Attributes
    ----------
    value : tuple[np.ndarray, np.ndarray, dict[str, Hashable]]
    """

    def __init__(self, parent_parameters, recovered_rotation_scale_phase=None):
        super().__init__(
            LogPolParams.RECOVERED_ROTATION_SCALE_PHASE,
            RecoveredRotationScalePhaseParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_rotation_scale_phase is not None:
            self.value = recovered_rotation_scale_phase

    @staticmethod
    def _value_function_handle(
        params: Dict[Enum, Parameter]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Hashable]]:
        return ImageMethods.recover_rs(
            params[LogPolParams.WARPED_FOURIER_REF_IMG].value,
            params[LogPolParams.WARPED_FOURIER_MOD_IMG].value,
            params[LogPolParams.REF_IMG].value.shape,
            params[LogPolParams.UPSAMPLING].value,
            params[LogPolParams.WINDOW_RADIUS_EXP].value,
        )


class RecoveredRotationParam(Parameter):
    """
    Recovered rotation angle and error in degrees.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much
    larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rotation=None):
        super().__init__(
            LogPolParams.RECOVERED_ROTATION,
            RecoveredRotationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_rotation is not None:
            self.value = recovered_rotation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        rotation, _, _ = params[LogPolParams.RECOVERED_ROTATION_SCALE_PHASE].value
        return cast(np.ndarray, rotation)


class RecoveredScaleParam(Parameter):
    """
    Recovered scaling factor and error in degrees.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much
    larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_scale=None):
        super().__init__(
            LogPolParams.RECOVERED_SCALE,
            RecoveredScaleParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_scale is not None:
            self.value = recovered_scale

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        _, scale, _ = params[LogPolParams.RECOVERED_ROTATION_SCALE_PHASE].value
        return cast(np.ndarray, scale)


class RecoveredRotScaleParam(ImageParameter):
    """
    Rotation and scaling recovered image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_scale_img=None):
        super().__init__(
            LogPolParams.RECOVERED_ROT_SCALE_IMG,
            RecoveredRotScaleParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Rotation Recovered Image"
        if recovered_rot_scale_img is not None:
            self.value = recovered_rot_scale_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[LogPolParams.MOD_IMG].value,
            angle=params[LogPolParams.RECOVERED_ROTATION].value[0],
            scale=params[LogPolParams.RECOVERED_SCALE].value[0],
            translation=(0, 0),
            inverse=True,
            preserve_range=True,
        )


# TODO rename angle, rotation such that angle in deg, rot in rad
# TODO possibly define seperate function in ImageMethods
class RecoveredTranslationParam(Parameter):
    """
    Recovered x,y translation vector and error.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much
    larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_translation=None):
        super().__init__(
            LogPolParams.RECOVERED_TRANSLATION,
            RecoveredTranslationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_translation is not None:
            self.value = recovered_translation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        shifts, error, _ = phase_cross_correlation(
            params[LogPolParams.REF_IMG].value,
            params[LogPolParams.RECOVERED_ROT_SCALE_IMG].value,
            upsample_factor=params[LogPolParams.UPSAMPLING].value,
        )
        tform = AffineTransform(
            scale=params[LogPolParams.RECOVERED_SCALE].value[0],
            rotation=np.deg2rad(-params[LogPolParams.RECOVERED_ROTATION].value[0]),
        )
        shifts = tform.params[:2, :2] @ shifts[::-1]
        return np.array([*shifts, error])


class RecoveredRotScaleTr(ImageParameter):
    """
    Rotation, scaling and translation recovered image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_scale_tr_img=None):
        super().__init__(
            LogPolParams.RECOVERED_ROT_SCALE_TR_IMG,
            RecoveredRotScaleTr._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Recovered Image"
        if recovered_rot_scale_tr_img is not None:
            self.value = recovered_rot_scale_tr_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[LogPolParams.MOD_IMG].value,
            angle=params[LogPolParams.RECOVERED_ROTATION].value[0],
            scale=params[LogPolParams.RECOVERED_SCALE].value[0],
            translation=params[LogPolParams.RECOVERED_TRANSLATION].value[:-1],
            inverse=True,
        )
