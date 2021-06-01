"""
The radon transform based image registration solver.

Author: Fabian A. Preiss
"""

from typing import Optional

import numpy as np

from imgreg.models.radon.enums import RadonParams
from imgreg.models.radon.params import (
    AngleSelectParam,
    AngularPrecisionParam,
    ExponentialFilterSignalNoiseParam,
    ModImageParam,
    ModRotationParam,
    RecoveredRotationParam,
    RecoveredRotParam,
    RecoveredRotTrParam,
    RecoveredTranslationParam,
    RefImageParam,
    RefRotationParam,
    RotationCandidateParam,
    ThetaParam,
    UpsamplingParam,
)
from imgreg.util.solver import Solver


class RadonSolver(Solver):
    r"""
    Implements an image registration model based on the radon transform.

    The model tries to reconstruct the difference of rotation and translation between two images.

    Parameters
    ----------
    ref_img : numpy.ndarray
        The original input image (one color channel only).
    mod_img : numpy.ndarray
        The modified input image (one color channel only).

    Notes
    -----
    The model implements the following dependency graph to construct it's `Parameter`\ s.

        .. graphviz:: ../_assets/dependencies_radon.gv

    The `Parameter`\ s are documented in :ref:`imgreg.models.radon.params`.

    Examples
    --------
    We can visualize the internal `ImageParameter`\ s of the model as follows:

    .. plot:: tutorial/pyplots/radonsolver.py
       :include-source:

    If we simply want to create a model and access the recovered values we first setup a model:

    >>> import numpy as np
    >>> import imgreg.data as data
    >>> from imgreg.models.radon import RadonSolver
    >>> ref_img = np.array(data.ref_img())
    >>> mod_img = np.array(data.mod_img())
    >>> ras = RadonSolver(ref_img, mod_img)

    Now the parameters of the model can now be accessed as follows:

    >>> ras.RECOVERED_ROTATION.value
    array([-11.35802469,   0.28284271])

    >>> ras.RECOVERED_TRANSLATION.value
    array([-11.80822224,  25.77936941,   0.30466765])
    """

    def __init__(
        self, ref_img: Optional[np.ndarray] = None, mod_img: Optional[np.ndarray] = None
    ):
        super().__init__()
        p_ref_img = RefImageParam(ref_img)
        p_mod_img = ModImageParam(mod_img)
        p_angular_precision = AngularPrecisionParam(0.2)
        p_exponential_filter_signal_noise = ExponentialFilterSignalNoiseParam(100)
        p_theta = ThetaParam(np.arange(0, 180, 1))
        p_upsampling = UpsamplingParam(10)
        p_ref_rotation = RefRotationParam(
            {p_ref_img, p_theta, p_exponential_filter_signal_noise, p_angular_precision}
        )
        p_mod_rotation = ModRotationParam(
            {p_mod_img, p_theta, p_exponential_filter_signal_noise, p_angular_precision}
        )
        p_rotation_candidate = RotationCandidateParam({p_ref_rotation, p_mod_rotation})
        p_angle_select = AngleSelectParam(
            {p_mod_img, p_ref_img, p_rotation_candidate, p_upsampling}
        )

        p_recovered_rotaton = RecoveredRotationParam(
            {p_rotation_candidate, p_angle_select}
        )
        p_recovered_rot = RecoveredRotParam({p_angle_select})
        p_recovered_translation = RecoveredTranslationParam({p_angle_select})
        p_recovered_rot_tr = RecoveredRotTrParam({p_angle_select})
        self._register_params(
            {
                p_ref_img,
                p_mod_img,
                p_angular_precision,
                p_exponential_filter_signal_noise,
                p_theta,
                p_upsampling,
                p_ref_rotation,
                p_mod_rotation,
                p_rotation_candidate,
                p_angle_select,
                p_recovered_rotaton,
                p_recovered_rot,
                p_recovered_translation,
                p_recovered_rot_tr,
            }
        )
        self._generate_dep_graphs()
        for key in RadonParams:
            setattr(self, key.name, self[key])


#
