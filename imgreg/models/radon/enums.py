"""
Enum of the radon parameters.

Author: Fabian A. Preiss
"""
from enum import Enum


class RadonParams(Enum):
    REF_IMG = "REF_IMG"
    """Reference image."""
    MOD_IMG = "MOD_IMG"
    """Modified image."""
    ANGULAR_PRECISION = "ANGULAR_PRECISION"
    """Targeted angular precision in degrees."""
    EXPONENTIAL_FILTER_SIGNAL_NOISE = "EXPONENTIAL_FILTER_SIGNAL_NOISE"
    """Signal to noise ratio when applying the min-max exponential filter."""
    THETA = "THETA"
    """The initial search angles."""
    UPSAMPLING = "UPSAMPLING"
    """Upsampling factor."""
    REF_ROTATION = "REF_ROTATION"
    """Recovered rotation angle and error in degrees of the reference image."""
    MOD_ROTATION = "MOD_ROTATION"
    """Recovered rotation angle and error in degrees of the modified image."""
    ROTATION_CANDIDATE = "ROTATION_CANDIDATE"
    """Candidate for the recovered rotation angle + error between the modified and reference image."""
    ANGLE_SELECT = "ANGLE_SELECT"
    """A solver model for angle selection."""
    RECOVERED_ROTATION = "RECOVERED_ROTATION"
    """The recovered rotation angle and error between the modified and reference image."""
    RECOVERED_ROT_IMG = "RECOVERED_ROT_IMG"
    """Rotation recovered image."""
    RECOVERED_TRANSLATION = "RECOVERED_TRANSLATION"
    """Recovered x,y translation vector and error."""
    RECOVERED_ROT_TR_IMG = "RECOVERED_ROT_TR_IMG"
    """Rotation, and translation recovered image."""
