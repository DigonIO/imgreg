from enum import Enum, auto
from typing import Any, Set, Tuple

import numpy as np
import pytest
from helpers import mod_image, ref_image

from imgreg.util.params import ImageParameter, Parameter, ParameterError
from imgreg.util.solver import Solver


class ParamID(Enum):
    PARAM_PARENT = auto()
    PARAM = auto()
    PARAM2 = auto()
    INT_PARAM = auto()
    FLOAT_PARAM = auto()
    STR_PARAM = auto()
    TUPLE_PARAM = auto()
    PARAM_CHILD = auto()


def supposed_int_returner(parameters: Set[Parameter]) -> int:
    return parameters[ParamID.INT_PARAM].value


def supposed_float_returner(parameters: Set[Parameter]) -> float:
    return parameters[ParamID.FLOAT_PARAM].value


def supposed_str_returner(parameters: Set[Parameter]) -> str:
    return parameters[ParamID.STR_PARAM].value


def supposed_tuple_returner(parameters: Set[Parameter]) -> Tuple[float, str]:
    return (
        parameters[ParamID.FLOAT_PARAM].value,
        parameters[ParamID.STR_PARAM].value,
    )


@pytest.fixture(
    params=[
        [0, supposed_int_returner, int],
        [1, supposed_float_returner, float],
        [2, supposed_str_returner, str],
        [3, supposed_tuple_returner, Tuple[float, str]],
    ],
)
def func(request):
    return request.param


class TestParameter:
    @pytest.mark.parametrize(
        "arg, valid",
        (
            [1, [True, True, False, False]],
            [7.3, [False, True, False, False]],
            ["abc", [False, False, True, False]],
            [(1.2, "abc"), [False, False, False, True]],
            [("abc", 1.2), [False, False, False, False]],
            [(1, 1.2, 1), [False, False, False, False]],
            [[1.2, "abc"], [False, False, False, False]],
        ),
    )
    def test_param_types(self, func, arg, valid):
        idx, handle, type_info = func
        param = Parameter(ParamID.PARAM, handle)
        assert param.enum_id == ParamID.PARAM
        with pytest.raises(ParameterError):
            param.value
        if not valid[idx]:
            with pytest.raises(ParameterError):
                param.value = arg
        else:
            param.value = arg
            assert param.value == arg

    def test_parent_child(self):
        p_float = Parameter(ParamID.FLOAT_PARAM, float)
        p_str = Parameter(ParamID.STR_PARAM, str)
        p_tuple_child = Parameter(
            ParamID.TUPLE_PARAM, supposed_tuple_returner, {p_float, p_str}
        )
        p_float.value = 1.1
        p_str.value = "abc"
        assert p_tuple_child.value == (1.1, "abc")
        assert len(p_tuple_child.parents) == 2
        assert len(p_float.parents) == len(p_str.parents) == 0
        assert len(p_tuple_child.children) == 0

        # parameters only know about their children in a solver environment
        p_float.value = 7.3
        assert p_tuple_child.value != (7.3, "abc")
        assert len(p_float.children) == len(p_str.children) != 1

        Solver({p_float, p_str, p_tuple_child})
        assert p_tuple_child.value == (1.1, "abc")
        p_float.value = 3.3
        assert p_tuple_child.value == (3.3, "abc")
        assert len(p_float.children) == len(p_str.children) == 1


class TestImgParameter:
    def test_simple_image(self, ref_image):
        simple_img = ImageParameter(ParamID.PARAM, np.ndarray)

        simple_img.title = "title"
        assert simple_img.title == "title"

        with pytest.raises(ParameterError):
            simple_img.display()

        simple_img.value = ref_image
        simple_img.display()
