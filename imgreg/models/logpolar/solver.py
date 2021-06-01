"""
The log-polar transform based image registration solver.

Author: Fabian A. Preiss
"""

from typing import Optional

import numpy as np

from imgreg.models.logpolar.enums import LogPolParams
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
from imgreg.util.solver import Solver


# NOTE: try applying exponential filter before Gaussdiff
class LogPolarSolver(Solver):
    r"""
    Implements an image registration model based on the log-polar transform.

    The model tries to reconstruct the difference of scale, rotation and
    translation between two images.

    Parameters
    ----------
    ref_img : numpy.ndarray
        The original input image (one color channel only).
    mod_img : numpy.ndarray
        The modified input image (one color channel only).

    Notes
    -----
    Build based on the approach of the example code [#f1]_ from scikit-image. Alternative
    implementations using feature based detection algorithms are shown in [#f2]_.

    The model implements the following dependency graph to construct it's `Parameter`\ s.

        .. graphviz:: ../_assets/dependencies_logpolar.gv

    The `Parameter`\ s are documented in :ref:`imgreg.models.logpolar.params`.

    References
    ----------
    .. [#f1] `Using Polar and Log-Polar Transformations for Registration
       <https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html#sphx-glr-auto-examples-registration-plot-register-rotation-py>`_
    .. [#f2] `ORB feature detector and binary descriptor
       <https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html#sphx-glr-auto-examples-features-detection-plot-orb-py>`_

    Examples
    --------
    We can visualize the internal `ImageParameter`\ s of the model as follows:

    .. plot:: tutorial/pyplots/logpolarsolver.py
       :include-source:

    If we simply want to create a model and access the recovered values we first setup a model:

    >>> import numpy as np
    >>> import imgreg.data as data
    >>> from imgreg.models.logpolar import LogPolarSolver, LogPolParams
    >>> ref_img = np.array(data.ref_img())
    >>> mod_img = np.array(data.mod_img())
    >>> lps = LogPolarSolver(ref_img, mod_img)

    Now the parameters of the model can now be accessed as follows:

    >>> lps.RECOVERED_ROTATION.value
    array([-13.06730769,   0.11259774])

    >>> lps.RECOVERED_TRANSLATION.value
    array([-17.98318062,  31.037803  ,   0.42407651])
    """

    def __init__(
        self, ref_img: Optional[np.ndarray] = None, mod_img: Optional[np.ndarray] = None
    ):
        super().__init__()

        p_ref_img = RefImageParam(ref_img)
        p_mod_img = ModImageParam(mod_img)
        p_gauss_diff = GaussDiffParam((5, 20))
        p_window_weight = WindowWeightParam(1)
        p_window_type = WindowTypeParam("hann")
        p_window_radius_exp = WindowRadiusExpParam(3)
        p_upsampling = UpsamplingParam(10)

        p_bounds = BoundsParam({p_ref_img, p_window_radius_exp})
        p_gauss_diff_ref_img = GaussDiffWindowRefParam(
            {p_ref_img, p_gauss_diff, p_window_weight, p_window_type}
        )
        p_gauss_diff_mod_img = GaussDiffWindowModParam(
            {p_mod_img, p_gauss_diff, p_window_weight, p_window_type}
        )
        p_fourier_ref_img = FourierRefParam({p_gauss_diff_ref_img, p_bounds})
        p_fourier_mod_img = FourierModParam({p_gauss_diff_mod_img, p_bounds})

        p_warped_fourier_ref_img = WarpedFourierRefParam(
            {p_fourier_ref_img, p_window_radius_exp}
        )
        p_warped_fourier_mod_img = WarpedFourierModParam(
            {p_fourier_mod_img, p_window_radius_exp}
        )
        p_recovered_rotation_scale_phase = RecoveredRotationScalePhaseParam(
            {
                p_warped_fourier_ref_img,
                p_warped_fourier_mod_img,
                p_ref_img,
                p_upsampling,
                p_window_radius_exp,
            }
        )
        p_recovered_rotation = RecoveredRotationParam(
            {p_recovered_rotation_scale_phase}
        )
        p_recovered_scale = RecoveredScaleParam({p_recovered_rotation_scale_phase})
        p_recovered_rot_scale_img = RecoveredRotScaleParam(
            {p_mod_img, p_recovered_rotation, p_recovered_scale}
        )
        p_recovered_translation = RecoveredTranslationParam(
            {
                p_ref_img,
                p_recovered_rot_scale_img,
                p_upsampling,
                p_recovered_rotation,
                p_recovered_scale,
            }
        )
        p_recovered_rot_scale_tr = RecoveredRotScaleTr(
            {
                p_mod_img,
                p_recovered_rotation,
                p_recovered_scale,
                p_recovered_translation,
            }
        )
        self._register_params(
            {
                p_ref_img,
                p_mod_img,
                p_gauss_diff,
                p_window_weight,
                p_window_type,
                p_window_radius_exp,
                p_upsampling,
                p_bounds,
                p_gauss_diff_ref_img,
                p_gauss_diff_mod_img,
                p_fourier_ref_img,
                p_fourier_mod_img,
                p_warped_fourier_ref_img,
                p_warped_fourier_mod_img,
                p_recovered_rotation_scale_phase,
                p_recovered_rotation,
                p_recovered_scale,
                p_recovered_rot_scale_img,
                p_recovered_translation,
                p_recovered_rot_scale_tr,
            }
        )
        self._generate_dep_graphs()
        for key in LogPolParams:
            setattr(self, key.name, self[key])
