"""
Parameter classes implementing lazy evaluation and dependency resolution.

Author: Fabian A. Preiss
"""

from __future__ import annotations

import types
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
from typeguard import check_type


class ParameterError(Exception):
    pass


class ParameterRegisterError(Exception):
    pass


def interface_function_handle(parameter: Dict[Enum, Parameter]) -> Any:
    r"""Construct value using parent `Parameter`\ s"""
    raise NotImplementedError


# TODO
# parents/children internal access: Dict[Enum, Parameter]
# parents/children external access: Set[Parameter]
class Parameter:
    r"""
    Base `Parameter` class, aware of related `Parameter`\ s and lazy evaluating.
    """

    def __init__(
        self,
        enum_id: Enum,
        value_definition: Union[Type[Any], Callable[[Dict[Enum, Parameter]], Any]],
        parent_parameters: Optional[Set[Parameter]] = None,
    ):
        self.__enum_id = enum_id
        self.__parents: Dict[Enum, Parameter] = (
            {}
            if parent_parameters is None
            else {parent.enum_id: parent for parent in parent_parameters}
        )
        # children have to be set from outside user
        self.__children: Dict[Enum, Parameter] = {}
        self.__descendants: Dict[Enum, Parameter] = {}
        self.__function_handle: Callable[[Dict[Enum, Parameter]], Any]
        self.__type_info: Union[Type[Any], Callable[[Dict[Enum, Parameter]], Any]]

        if isinstance(value_definition, types.FunctionType):
            self.__type_info = value_definition.__annotations__.get("return", Any)
            self.__function_handle = value_definition
        else:
            try:
                self.__type_info = value_definition
            except TypeError as err:
                raise ParameterError(
                    "value_definition is neither a type or a function."
                ) from err
            self.__function_handle = interface_function_handle
        if self.__type_info is Any:
            raise ParameterError(
                "value_definition requires type informations other than Any."
            )
        self.__value: Any = None
        self.__const: bool = False

    def __gt__(self, other: Parameter) -> bool:
        if type(self).__name__ > type(other).__name__:
            return True
        if type(self).__name__ < type(other).__name__:
            return False
        return self.enum_id.name > other.enum_id.name

    def __lt__(self, other: Parameter) -> bool:
        if type(self).__name__ > type(other).__name__:
            return False
        if type(self).__name__ < type(other).__name__:
            return True
        return self.enum_id.name < other.enum_id.name

    @property
    def enum_id(self) -> Enum:
        """Returns the Parameter ID as an enumeration."""
        return self.__enum_id

    @property
    def type_info(self):
        """The type of this `Parameter`."""
        return self.__type_info

    @property
    def const(self) -> bool:
        """Flag if the value can be overwritten once initialized."""
        return self.__const

    @const.setter
    def const(self, const: bool) -> None:
        if self.__const:
            raise ParameterError("Parameter is already flagged as constant.")
        self.__const = const

    @property
    def value(self):
        """The value of this `Parameter`."""
        if self.__value is None:
            self.__clear_descendants()
            try:
                self.__value = self.__function_handle(self.__parents)
            except (NotImplementedError, KeyError) as err:
                raise ParameterError(err)
        return self.__value

    @value.setter
    def value(self, value: Any):
        self.__clear_descendants()
        try:
            check_type("value", value, self.__type_info)
        except TypeError as err:
            raise ParameterError(
                "interface_function_handle annotation does not match given value"
            ) from err
        if self.__const and self.__value is not None:
            raise ParameterError("Cannot overwrite constant parameter.")
        self.__value = value

    def clear(self, clear_descendants=True) -> None:
        r"""Clear the value stored in this and dependent `Parameter`\ s."""
        if clear_descendants:
            self.__clear_descendants()
        if not self.__const:
            self.__value = None

    @property
    def parents(self) -> Dict[Enum, Parameter]:
        r"""Parent `Parameter`\ s of this `Parameter`\ s instance."""
        return self.__parents.copy()

    @property
    def children(self) -> Dict[Enum, Parameter]:
        r"""`Parameter`\ s with a child relation to this `Parameter`\ s instance."""
        return self.__children.copy()

    def add_child(self, child: Parameter) -> None:
        r"""Assign a Parameter a child relation"""
        if child.enum_id in self.__parents:
            raise ParameterError(
                "Neighbour can't be parent and child at the same time."
            )
        self.__children[child.enum_id] = child

    @property
    def descendants(self) -> Dict[Enum, Parameter]:
        r"""`Parameter`\ s with a descendant relation to this `Parameter`\ s instance."""
        return self.__descendants.copy()

    def add_descendant(self, descendant: Parameter) -> None:
        r"""Assign a Parameter a child relation"""
        if descendant.enum_id in self.__parents:
            raise ParameterError(
                f"Parameter {descendant.enum_id} can't be parent and descendant at the same time."
            )
        self.__descendants[descendant.enum_id] = descendant

    def __clear_descendants(self) -> None:
        for descendant_enum_id in self.__descendants:
            self.__descendants[descendant_enum_id].clear(clear_descendants=False)


class ImageParameter(Parameter):
    """`Parameter` with additional display functionality for stored image."""

    def __init__(
        self,
        enum_id: Enum,
        value_definition: Union[Type[Any], Callable[[Dict[Enum, Parameter]], Any]],
        parent_parameters: Optional[Set[Parameter]] = None,
    ):
        super().__init__(enum_id, value_definition, parent_parameters)
        self.__title = None
        self.__bounds: Optional[Tuple[int, int, int, int]] = None
        self.__bounds_lookup: Optional[Enum] = None
        self.__cmap = "gray"
        self.__aspect: Optional[float] = None

    def display(self, axsp: Any = None):
        """
        Display the stored image using matplotlib.

        Parameters
        ----------
        aspx : matplotlib.axes._subplots.AxesSubplot
            AxesSubplot on which to draw

        Notes
        -----
        Uses the `title`, `cmap` and `bounds` properties of the `ImageParameter` class.
        """
        if axsp is None:
            fig = plt.figure()
            axsp = fig.subplots()
        if self.__title is not None:
            axsp.set_title(self.__title)
        if self.bounds is None:
            axsp.imshow(self.value, cmap=self.__cmap, aspect=self.__aspect)
        else:
            (r_lower, r_upper, c_lower, c_upper) = self.bounds
            axsp.imshow(
                self.value[r_lower:r_upper, c_lower:c_upper],
                cmap=self.__cmap,
                aspect=self.__aspect,
            )

    @property
    def title(self) -> Optional[str]:
        """The image title used by the display function."""
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title

    @property
    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Cropping boundaries for the display function."""
        if self.__bounds is None and self.__bounds_lookup is None:
            return None
        if self.__bounds is None:
            return cast(
                Optional[Tuple[int, int, int, int]],
                self.parents[cast(Enum, self.__bounds_lookup)].value,
            )
        return self.__bounds

    @bounds.setter
    def bounds(self, bounds: Tuple[int, int, int, int]):
        self.__bounds = bounds

    @property
    def cmap(self) -> str:
        """The colormap used by matplotlib."""
        return self.__cmap

    @cmap.setter
    def cmap(self, cmap: str):
        self.__cmap = cmap

    @property
    def aspect(self) -> Optional[float]:
        """The aspect ratio used by matplotlib."""
        return self.__aspect

    @aspect.setter
    def aspect(self, aspect: float):
        self.__aspect = aspect

    def set_bounds_lookup(self, bounds_enum_id: Enum):
        """Reference a boundary Parameter using its `enum_id` as a lookup for cropping."""
        self.__bounds_lookup = bounds_enum_id
