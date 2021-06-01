"""
Module implementing the parameter classes for the Radon Model

Author: Fabian A. Preiss
"""

from enum import Enum
from typing import Dict, Optional, cast

import numpy as np
from skimage.registration import phase_cross_correlation

from imgreg.models.angleselect import AngleSelect, AngleSelectParams
from imgreg.models.radon.enums import RadonParams
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
        super().__init__(RadonParams.REF_IMG, np.ndarray)
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
        super().__init__(RadonParams.MOD_IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Modified Image"


class AngularPrecisionParam(Parameter):
    """
    Targeted angular precision in degrees.

    Notes
    -----
    The targeted error is the lower estimate under ideal assumptions, in practise won't be achieavable.

    Attributes
    ----------
    value : float
    """

    def __init__(self, angular_precision: float):
        super().__init__(RadonParams.ANGULAR_PRECISION, float)
        self.value = angular_precision


class ExponentialFilterSignalNoiseParam(Parameter):
    """
    Signal to noise ratio when applying the min-max exponential filter.

    Attributes
    ----------
    value : float
    """

    def __init__(self, exponential_filter_signal_noise: float):
        super().__init__(RadonParams.EXPONENTIAL_FILTER_SIGNAL_NOISE, float)
        self.value = exponential_filter_signal_noise


class ThetaParam(Parameter):
    """
    The initial search angles.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, theta: np.ndarray):
        super().__init__(RadonParams.THETA, np.ndarray)
        self.value = theta


class UpsamplingParam(Parameter):
    """
    Upsampling factor. 1 => no upsampling, 20 => precision to 1/20 of a pixel.

    Attributes
    ----------
    value : int
    """

    def __init__(self, upsampling: int):
        super().__init__(RadonParams.UPSAMPLING, int)
        self.value = upsampling


# derived properties


class RefRotationParam(Parameter):
    """
    Recovered rotation angle and error in degrees of the reference image.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much larger depending on the data.

    Attributes
    ----------
    value : float
    """

    def __init__(self, parent_parameters, ref_rotation: Optional[float] = None):
        super().__init__(
            RadonParams.REF_ROTATION,
            RefRotationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if ref_rotation is not None:
            self.value = ref_rotation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return np.array(
            [
                ImageMethods.max_sinogram_angle(
                    params[RadonParams.REF_IMG].value,
                    theta=params[RadonParams.THETA].value,
                    exp_filter_val=params[
                        RadonParams.EXPONENTIAL_FILTER_SIGNAL_NOISE
                    ].value,
                    precision=params[RadonParams.ANGULAR_PRECISION].value,
                ),
                params[RadonParams.ANGULAR_PRECISION].value,
            ]
        )


class ModRotationParam(Parameter):
    """
    Recovered rotation angle and error in degrees of the modified image.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, mod_rotation: Optional[float] = None):
        super().__init__(
            RadonParams.MOD_ROTATION,
            ModRotationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if mod_rotation is not None:
            self.value = mod_rotation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return np.array(
            [
                ImageMethods.max_sinogram_angle(
                    params[RadonParams.MOD_IMG].value,
                    theta=params[RadonParams.THETA].value,
                    exp_filter_val=params[
                        RadonParams.EXPONENTIAL_FILTER_SIGNAL_NOISE
                    ].value,
                    precision=params[RadonParams.ANGULAR_PRECISION].value,
                ),
                params[RadonParams.ANGULAR_PRECISION].value,
            ]
        )


class RotationCandidateParam(Parameter):
    """
    Candidate for the recovered rotation angle + error between the modified and reference image.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(
        self, parent_parameters, rotation_candidate: Optional[np.ndarray] = None
    ):
        super().__init__(
            RadonParams.ROTATION_CANDIDATE,
            RotationCandidateParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if rotation_candidate is not None:
            self.value = rotation_candidate

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return np.array(
            [
                params[RadonParams.MOD_ROTATION].value[0]
                - params[RadonParams.REF_ROTATION].value[0],
                (
                    params[RadonParams.MOD_ROTATION].value[1] ** 2
                    + params[RadonParams.REF_ROTATION].value[1] ** 2
                )
                ** 0.5,
            ]
        )


class AngleSelectParam(Parameter):
    """A solver model for angle selection."""

    def __init__(self, parent_parameters, angle_select: Optional[AngleSelect] = None):
        super().__init__(
            RadonParams.ANGLE_SELECT,
            AngleSelectParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if angle_select is not None:
            self.value = angle_select

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> AngleSelect:
        return AngleSelect(
            params[RadonParams.MOD_IMG].value,
            params[RadonParams.REF_IMG].value,
            params[RadonParams.ROTATION_CANDIDATE].value[0],
            params[RadonParams.ROTATION_CANDIDATE].value[0] + 180,
            params[RadonParams.UPSAMPLING].value,
        )


class RecoveredRotationParam(Parameter):
    """
    The recovered rotation angle and error between the modified and reference image.

    Notes
    -----
    The errors are a lower estimate under ideal assumptions and can be much larger depending on the data.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, angle_select: Optional[np.ndarray] = None):
        super().__init__(
            RadonParams.RECOVERED_ROTATION,
            RecoveredRotationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if angle_select is not None:
            self.value = angle_select

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        matching_angle = params[RadonParams.ROTATION_CANDIDATE].value.copy()
        angle_select = params[RadonParams.ANGLE_SELECT].value
        matching_angle[0] = angle_select[AngleSelectParams.RECOVERED_ROTATION].value
        return cast(np.ndarray, matching_angle)


class RecoveredRotParam(ImageParameter):
    """
    Rotation recovered image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_img=None):
        super().__init__(
            RadonParams.RECOVERED_ROT_IMG,
            RecoveredRotParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Rotation Recovered Image"
        if recovered_rot_img is not None:
            self.value = recovered_rot_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        angle_select = params[RadonParams.ANGLE_SELECT].value
        return cast(np.ndarray, angle_select[AngleSelectParams.RECOVERED_ROT_IMG].value)


class RecoveredTranslationParam(Parameter):
    """
    Recovered x,y translation vector and error.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_translation=None):
        super().__init__(
            RadonParams.RECOVERED_TRANSLATION,
            RecoveredTranslationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_translation is not None:
            self.value = recovered_translation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        angle_select = params[RadonParams.ANGLE_SELECT].value
        return cast(
            np.ndarray, angle_select[AngleSelectParams.RECOVERED_TRANSLATION].value
        )


class RecoveredRotTrParam(ImageParameter):
    """
    Rotation, and translation recovered image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_tr_img=None):
        super().__init__(
            RadonParams.RECOVERED_ROT_TR_IMG,
            RecoveredRotTrParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Recovered Image"
        if recovered_rot_tr_img is not None:
            self.value = recovered_rot_tr_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        angle_select = params[RadonParams.ANGLE_SELECT].value
        return cast(
            np.ndarray, angle_select[AngleSelectParams.RECOVERED_ROT_TR_IMG].value
        )
