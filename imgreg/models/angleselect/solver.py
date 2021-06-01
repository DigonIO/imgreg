"""
Angle selection matching based on image matching.

Author: Fabian A. Preiss
"""

import numpy as np

from imgreg.models.angleselect.enums import AngleSelectParams
from imgreg.models.angleselect.params import (
    AngleAParam,
    AngleBParam,
    ImageParam,
    RecoveredRotationParam,
    RecoveredRotParam,
    RecoveredRotTrParam,
    RecoveredTranslationParam,
    RefImageParam,
    RotAParam,
    RotBParam,
    RotTrAParam,
    RotTrBParam,
    SelectorParam,
    TranslationAParam,
    TranslationBParam,
    UpsamplingParam,
)
from imgreg.util.solver import Solver


class AngleSelect(Solver):
    r"""
    Implements a model for angle selection

    Parameters
    ----------
    img : numpy.ndarray
        The input image (one color channel only).
    ref_img : numpy.ndarray
        The reference image (one color channel only).
    angle_a, angle_b : float
        Candidate angles A and B in degrees
    upsampling : int
        Upsampling factor.

        1 => no upsampling, 20 => precision to 1/20 of a pixel.

    Notes
    -----

    The model implements the following dependency graph to construct it's `Parameter`\ s.

        .. graphviz:: ../_assets/dependencies_angleselect.gv
    """

    def __init__(
        self,
        img: np.ndarray,
        ref_img: np.ndarray,
        angle_a: float,
        angle_b: float,
        upsampling: int = 10,
    ):
        super().__init__()
        p_img = ImageParam(img)
        p_ref_img = RefImageParam(ref_img)
        p_angle_a = AngleAParam(angle_a)
        p_angle_b = AngleBParam(angle_b)
        p_upsampling = UpsamplingParam(upsampling)
        p_rot_a_img = RotAParam({p_img, p_angle_a})
        p_rot_b_img = RotBParam({p_img, p_angle_b})
        p_translation_a = TranslationAParam(
            {p_ref_img, p_rot_a_img, p_upsampling, p_angle_a}
        )
        p_translation_b = TranslationBParam(
            {p_ref_img, p_rot_b_img, p_upsampling, p_angle_b}
        )
        p_rot_tr_a_img = RotTrAParam({p_img, p_angle_a, p_translation_a})
        p_rot_tr_b_img = RotTrBParam({p_img, p_angle_b, p_translation_b})
        p_selector = SelectorParam({p_rot_tr_a_img, p_rot_tr_b_img, p_ref_img})
        p_recovered_rotation = RecoveredRotationParam(
            {p_selector, p_angle_a, p_angle_b}
        )
        p_recovered_rot_img = RecoveredRotParam({p_selector, p_rot_a_img, p_rot_b_img})
        p_recovered_translation = RecoveredTranslationParam(
            {p_selector, p_translation_a, p_translation_b}
        )
        p_recovered_rot_tr_img = RecoveredRotTrParam(
            {p_selector, p_rot_tr_a_img, p_rot_tr_b_img}
        )
        self._register_params(
            {
                p_img,
                p_ref_img,
                p_angle_a,
                p_angle_b,
                p_upsampling,
                p_rot_a_img,
                p_rot_b_img,
                p_translation_a,
                p_translation_b,
                p_rot_tr_a_img,
                p_rot_tr_b_img,
                p_selector,
                p_recovered_rotation,
                p_recovered_rot_img,
                p_recovered_translation,
                p_recovered_rot_tr_img,
            }
        )
        self._generate_dep_graphs()
        for key in AngleSelectParams:
            setattr(self, key.name, self[key])
