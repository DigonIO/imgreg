from enum import Enum, auto
from typing import Any, Set

import numpy as np
import pytest
from PIL import Image
from helpers import mod_image, ref_image

from imgreg.util.params import ImageParameter
from imgreg.util.solver import Solver


class ParamID(Enum):
    PIMG_0 = auto()
    PIMG_1 = auto()
    PIMG_2 = auto()


@pytest.fixture(
    params=[
        [
            [
                ImageParameter(ParamID.PIMG_0, np.ndarray),
                ImageParameter(ParamID.PIMG_1, np.ndarray),
            ],
        ],
        [
            [
                ImageParameter(ParamID.PIMG_0, np.ndarray),
                ImageParameter(ParamID.PIMG_1, np.ndarray),
                ImageParameter(ParamID.PIMG_2, np.ndarray),
            ],
        ],
    ]
)
def func(request, ref_image):
    for param in request.param[0]:
        param.value = ref_image
    return request.param


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
def test_display(func):
    solver = Solver(set(func[0]))
    solver.display([param.enum_id for param in func[0]])


# TODO while line 67 will not be exec in the solver init?
def test_dep_graph(func):
    solver = Solver(set(func[0]))
    solver._get_dep_graph()
