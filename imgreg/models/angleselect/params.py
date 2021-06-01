"""
Module implementing the parameter classes for the angle selector Model

Author: Fabian A. Preiss
"""
from enum import Enum
from typing import Dict, Optional, cast

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform

from imgreg.models.angleselect.enums import AngleSelectParams
from imgreg.util.methods import ImageMethods
from imgreg.util.params import ImageParameter, Parameter


class ImageParam(ImageParameter):
    """
    Image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        super().__init__(AngleSelectParams.IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Image"


class RefImageParam(ImageParameter):
    """
    Reference image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        super().__init__(AngleSelectParams.REF_IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Reference Image"


class AngleAParam(Parameter):
    """
    Candidate Angle A.

    Attributes
    ----------
    value : float
    """

    def __init__(self, angle_a: float):
        super().__init__(AngleSelectParams.ANGLE_A, float)
        self.value = angle_a


class AngleBParam(Parameter):
    """
    Candidate Angle B.

    Attributes
    ----------
    value : float
    """

    def __init__(self, angle_b: float):
        super().__init__(AngleSelectParams.ANGLE_B, float)
        self.value = angle_b


class UpsamplingParam(Parameter):
    """
    Upsampling factor.

    1 => no upsampling, 20 => precision to 1/20 of a pixel.

    Attributes
    ----------
    value : int
    """

    def __init__(self, upsampling: int):
        super().__init__(AngleSelectParams.UPSAMPLING, int)
        self.value = upsampling


# derived properties


class RotAParam(ImageParameter):
    """
    Image Rotated by Angle A.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_img=None):
        super().__init__(
            AngleSelectParams.ROT_A_IMG,
            RotAParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Image Rotated A"
        if recovered_rot_img is not None:
            self.value = recovered_rot_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[AngleSelectParams.IMG].value,
            angle=params[AngleSelectParams.ANGLE_A].value,
            scale=1,
            translation=(0, 0),
            inverse=True,
            preserve_range=True,
        )


class RotBParam(ImageParameter):
    """
    Image Rotated by Angle B.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_img=None):
        super().__init__(
            AngleSelectParams.ROT_B_IMG,
            RotBParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Image Rotated B"
        if recovered_rot_img is not None:
            self.value = recovered_rot_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[AngleSelectParams.IMG].value,
            angle=params[AngleSelectParams.ANGLE_B].value,
            scale=1,
            translation=(0, 0),
            inverse=True,
            preserve_range=True,
        )


class TranslationAParam(Parameter):
    """
    Recovered x,y translation vector and error for A.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_translation=None):
        super().__init__(
            AngleSelectParams.TRANSLATION_A,
            TranslationAParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_translation is not None:
            self.value = recovered_translation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        shifts, error, _ = phase_cross_correlation(
            params[AngleSelectParams.REF_IMG].value,
            params[AngleSelectParams.ROT_A_IMG].value,
            upsample_factor=params[AngleSelectParams.UPSAMPLING].value,
        )
        tform = AffineTransform(
            scale=1,
            rotation=np.deg2rad(-params[AngleSelectParams.ANGLE_A].value),
        )
        shifts = tform.params[:2, :2] @ shifts[::-1]
        return np.array([*shifts, error])


class TranslationBParam(Parameter):
    """
    Recovered x,y translation vector and error for B.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_translation=None):
        super().__init__(
            AngleSelectParams.TRANSLATION_B,
            TranslationBParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_translation is not None:
            self.value = recovered_translation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        shifts, error, _ = phase_cross_correlation(
            params[AngleSelectParams.REF_IMG].value,
            params[AngleSelectParams.ROT_B_IMG].value,
            upsample_factor=params[AngleSelectParams.UPSAMPLING].value,
        )
        tform = AffineTransform(
            scale=1,
            rotation=np.deg2rad(-params[AngleSelectParams.ANGLE_B].value),
        )
        shifts = tform.params[:2, :2] @ shifts[::-1]
        return np.array([*shifts, error])


class RotTrAParam(ImageParameter):
    """
    Rotation, and translation recovered image A.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_tr_img=None):
        super().__init__(
            AngleSelectParams.ROT_TR_A_IMG,
            RotTrAParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Recovered Image"
        if recovered_rot_tr_img is not None:
            self.value = recovered_rot_tr_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[AngleSelectParams.IMG].value,
            angle=params[AngleSelectParams.ANGLE_A].value,
            scale=1,
            translation=params[AngleSelectParams.TRANSLATION_A].value[:-1],
            inverse=True,
            preserve_range=True,
        )


class RotTrBParam(ImageParameter):
    """
    Rotation, and translation recovered image B.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_tr_img=None):
        super().__init__(
            AngleSelectParams.ROT_TR_B_IMG,
            RotTrBParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "Recovered Image"
        if recovered_rot_tr_img is not None:
            self.value = recovered_rot_tr_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.compute_rts(
            params[AngleSelectParams.IMG].value,
            angle=params[AngleSelectParams.ANGLE_B].value,
            scale=1,
            translation=params[AngleSelectParams.TRANSLATION_B].value[:-1],
            inverse=True,
            preserve_range=True,
        )


class SelectorParam(Parameter):
    """
    True, if A matches better than B

    Attributes
    ----------
    value : bool
    """

    def __init__(self, parent_parameters, selection=None):
        super().__init__(
            AngleSelectParams.SELECTOR,
            SelectorParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if selection is not None:
            self.value = selection

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> bool:
        return ImageMethods.norm_rel_l2(
            params[AngleSelectParams.ROT_TR_A_IMG].value,
            params[AngleSelectParams.REF_IMG].value,
        ) <= ImageMethods.norm_rel_l2(
            params[AngleSelectParams.ROT_TR_B_IMG].value,
            params[AngleSelectParams.REF_IMG].value,
        )


class RecoveredRotationParam(Parameter):
    """
    Recovered rotation angle and error between the modified and reference image.

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
            AngleSelectParams.RECOVERED_ROTATION,
            RecoveredRotationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_rotation is not None:
            self.value = recovered_rotation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> float:
        return cast(
            float,
            params[AngleSelectParams.ANGLE_A].value
            if params[AngleSelectParams.SELECTOR].value
            else params[AngleSelectParams.ANGLE_B].value,
        )


class RecoveredRotParam(ImageParameter):
    """
    Rotation recovered image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_rot_img=None):
        super().__init__(
            AngleSelectParams.RECOVERED_ROT_IMG,
            RecoveredRotParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_rot_img is not None:
            self.value = recovered_rot_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return cast(
            np.ndarray,
            params[AngleSelectParams.ROT_A_IMG].value
            if params[AngleSelectParams.SELECTOR].value
            else params[AngleSelectParams.ROT_B_IMG].value,
        )


class RecoveredTranslationParam(Parameter):
    """
    Recovered x,y translation vector and error.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters, recovered_translation=None):
        super().__init__(
            AngleSelectParams.RECOVERED_TRANSLATION,
            RecoveredTranslationParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_translation is not None:
            self.value = recovered_translation

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return cast(
            np.ndarray,
            params[AngleSelectParams.TRANSLATION_A].value
            if params[AngleSelectParams.SELECTOR].value
            else params[AngleSelectParams.TRANSLATION_B].value,
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
            AngleSelectParams.RECOVERED_ROT_TR_IMG,
            RecoveredRotTrParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        if recovered_rot_tr_img is not None:
            self.value = recovered_rot_tr_img

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return cast(
            np.ndarray,
            params[AngleSelectParams.ROT_TR_A_IMG].value
            if params[AngleSelectParams.SELECTOR].value
            else params[AngleSelectParams.ROT_TR_B_IMG].value,
        )
