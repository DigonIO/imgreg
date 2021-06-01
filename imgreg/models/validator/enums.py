"""
Enum of the validator.

Author: Fabian A. Preiss
"""
from enum import Enum


class ValidatorParams(Enum):
    IMG = "IMG"
    """Image."""
    REF_IMG = "REF_IMG"
    """Reference image."""
    NORM_REL_L2 = "NORM_REL_L2"
    """Relative L2 Norm similarity measurement for the images."""
    ABSOLUTE_DIFFERENCE_IMG = "ABSOLUTE_DIFFERENCE_IMG"
    """Absolute value of the difference between the images."""
    SQUARED_DIFFERENCE_IMG = "SQUARED_DIFFERENCE_IMG"
    """Squared difference between the images."""
