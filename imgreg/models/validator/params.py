"""
Module implementing the parameter classes for a validator.

Author: Fabian A. Preiss
"""

from enum import Enum
from typing import Dict, Optional

import numpy as np

from imgreg.models.validator.enums import ValidatorParams
from imgreg.util.methods import ImageMethods
from imgreg.util.params import ImageParameter, Parameter

# Root Parameters


class ImageParam(ImageParameter):
    """
    Image.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        super().__init__(ValidatorParams.IMG, np.ndarray)
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
        super().__init__(ValidatorParams.REF_IMG, np.ndarray)
        if image is not None:
            self.value = image
        self.title = "Reference Image"


# derived properties


class NormRelL2Param(Parameter):
    """
    Relative L2 Norm similarity measurement for the images.

    Attributes
    ----------
    value : float
    """

    def __init__(self, parent_parameters):
        super().__init__(
            ValidatorParams.NORM_REL_L2,
            NormRelL2Param._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "norm(ref_image-image)/norm(ref_image)"

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> float:
        return ImageMethods.norm_rel_l2(
            params[ValidatorParams.IMG].value,
            params[ValidatorParams.REF_IMG].value,
        )


class AbsoluteDifferenceParam(ImageParameter):
    """
    Absolute value of the difference between the images.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters):
        super().__init__(
            ValidatorParams.ABSOLUTE_DIFFERENCE_IMG,
            AbsoluteDifferenceParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "abs(ref_image-image)"

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.abs_diff(
            params[ValidatorParams.IMG].value,
            params[ValidatorParams.REF_IMG].value,
        )


class SquaredDifferenceParam(ImageParameter):
    """
    Squared difference between the images.

    Attributes
    ----------
    value : numpy.ndarray
    """

    def __init__(self, parent_parameters):
        super().__init__(
            ValidatorParams.SQUARED_DIFFERENCE_IMG,
            SquaredDifferenceParam._value_function_handle,
            parent_parameters=parent_parameters,
        )
        self.title = "(ref_image-image)^2"

    @staticmethod
    def _value_function_handle(params: Dict[Enum, Parameter]) -> np.ndarray:
        return ImageMethods.sqr_diff(
            params[ValidatorParams.IMG].value,
            params[ValidatorParams.REF_IMG].value,
        )
