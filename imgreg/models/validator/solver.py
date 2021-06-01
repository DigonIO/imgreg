"""
A validator for image similarity.

Author: Fabian A. Preiss
"""

from typing import Optional

import numpy as np

from imgreg.models.validator.enums import ValidatorParams
from imgreg.models.validator.params import (
    AbsoluteDifferenceParam,
    ImageParam,
    NormRelL2Param,
    RefImageParam,
    SquaredDifferenceParam,
)
from imgreg.util.solver import Solver


class Validator(Solver):
    r"""
    Implements a validator model for image comparison.

    Parameters
    ----------
    img : numpy.ndarray
        The input image (one color channel only).
    ref_img : numpy.ndarray
        The reference image (one color channel only).

    Notes
    -----
    The model implements the following dependency graph to construct it's `Parameter`\ s.

        .. graphviz:: ../_assets/dependencies_validator.gv

    The `Parameter`\ s are documented in :ref:`imgreg.models.validator.params`.

    Examples
    --------
    We can visualize the internal `ImageParameter`\ s of the model as follows:

    .. plot:: tutorial/pyplots/validator.py
       :include-source:

    If we simply want to create a model and access the recovered values we first setup a model:

    >>> import numpy as np
    >>> import imgreg.data as data
    >>> from imgreg.models.validator import Validator
    >>> from imgreg.util.methods import ImageMethods
    >>> ref_img = np.array(data.ref_img())
    >>> img = ImageMethods.compute_rts(ref_img, angle=2, translation=(6,2))
    >>> val = Validator(img, ref_img)

    Calculate the relative norm of difference between the images.

    >>> val.NORM_REL_L2.value # doctest:+ELLIPSIS
    0.4875079942792...

    Note how this value approaches zero, as the image increase in their overlap:

    >>> val.IMG.value = ImageMethods.compute_rts(ref_img, angle=1, translation=(1,2))
    >>> val[ValidatorParams.NORM_REL_L2].value # doctest:+ELLIPSIS
    0.3942652180108...
    """

    def __init__(
        self, img: Optional[np.ndarray] = None, ref_img: Optional[np.ndarray] = None
    ):
        super().__init__()
        p_img = ImageParam(img)
        p_ref_img = RefImageParam(ref_img)
        p_norm_rel_l2 = NormRelL2Param({p_img, p_ref_img})
        p_absolute_difference = AbsoluteDifferenceParam({p_img, p_ref_img})
        p_squared_difference = SquaredDifferenceParam({p_img, p_ref_img})
        self._register_params(
            {
                p_img,
                p_ref_img,
                p_norm_rel_l2,
                p_absolute_difference,
                p_absolute_difference,
                p_squared_difference,
            }
        )
        self._generate_dep_graphs()
        for key in ValidatorParams:
            setattr(self, key.name, self[key])
