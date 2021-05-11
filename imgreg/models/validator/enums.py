"""
Enum of the validator.

Author: Fabian A. Preiss
"""
from enum import Enum


class ValidatorParams(Enum):
    IMG = "IMG"
    REF_IMG = "REF_IMG"
    NORM_REL_L2 = "NORM_REL_L2"
    ABSOLUTE_DIFFERENCE_IMG = "ABSOLUTE_DIFFERENCE_IMG"
    SQUARED_DIFFERENCE_IMG = "SQUARED_DIFFERENCE_IMG"
