"""
Solvermodel primitives.

Author: Fabian A. Preiss
"""
from __future__ import annotations

from enum import Enum
from typing import (
    Callable,
    Dict,
    Hashable,
    ItemsView,
    Iterable,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from graphviz import Digraph

from imgreg.util.graph import DAGraph
from imgreg.util.params import ImageParameter, Parameter


class SolverError(Exception):
    pass


def dependency_graph(parameters: Set[Parameter], invert=False) -> DAGraph:
    """Construct a `DAGraph` dependency graph given a `Parameter` set."""
    dep_graph = {}
    for parameter in parameters:
        if not invert:
            dep_graph[parameter] = {parent for parent in parameter.parents.values()}
        else:
            dep_graph[parameter] = {child for child in parameter.children.values()}
    return DAGraph(cast(Dict[Hashable, Set[Hashable]], dep_graph))


def dot_shape_func(parameter: Parameter) -> Dict[str, str]:
    """
    Generate the shape argument for a dot graph depending on the given Parameter

    Parameters
    ----------
    parameter : Parameter
        The parameter of a node
    """
    if isinstance(parameter, ImageParameter):
        shape = "box"
    else:
        shape = "oval"
    return {"shape": shape}


class Solver:
    r"""
    Interface for a Solvermodel constructed from a set of parameters.

    Constructs a dependency graph for the `Parameter`\ s of the model and allows for
    lazy evaluation of the `Parameter`\ s.

    Notes
    -----
    The state of this class is cached, when a parameter is changed on which later states depend, the
    properties ascendant to said parameter are removed from the cache. `Solver._get_dep_graph()`
    allows access to the internal dependency graph representation.
    """

    def __init__(self, parameters: Optional[Set[Parameter]] = None):
        self.__params: Dict[Enum, Parameter] = dict()
        self.__dependency_graph: Optional[DAGraph] = None
        self.__idependency_graph: Optional[DAGraph] = None
        if parameters is not None:
            self._register_params(parameters)
            self._generate_dep_graphs()
            for param in parameters:
                setattr(self, param.enum_id.name, self[param.enum_id])

    # TODO in concrete solver unittest if all Parameter objects are registered
    def __getitem__(self, item: Enum) -> Parameter:
        return self.__params[item]

    def _register_params(self, params: Iterable[Parameter]) -> None:
        for param in params:
            self._register_param(param)

    def _register_param(self, param: Parameter) -> None:
        if param.enum_id in self.__params:
            raise SolverError(
                f"Cannot register parameter '{param.enum_id.value}' two times."
            )
        self.__params[param.enum_id] = param
        for parent in param.parents.values():
            parent.add_child(param)

    # TODO insert descendants into each parameter here, then replace add_child methods etc.
    def _generate_dep_graphs(self) -> None:
        self.__dependency_graph = dependency_graph(
            set(self.__params.values()), invert=False
        )
        self.__idependency_graph = dependency_graph(
            set(self.__params.values()), invert=True
        )
        for (parameter, ascendants,) in cast(
            ItemsView[Parameter, Set[Parameter]],
            self.__idependency_graph.vertex_ascendants_dict.items(),
        ):
            for ascendant in ascendants:
                parameter.add_descendant(ascendant)

    def _get_dep_graph(self, invert=False) -> DAGraph:
        if self.__dependency_graph is None:
            self._generate_dep_graphs()

        result = self.__idependency_graph if invert else self.__dependency_graph
        return cast(DAGraph, result)

    def display(
        self,
        param_list: Sequence[Union[Enum, ImageParameter]],
        title: Optional[str] = None,
    ) -> None:
        """
        Fancy plot functionality for registered ImageParameters.

        Parameters
        ----------
        plotlist : sequence
            sequence, to access the ImageParameters registered in solver
        title : str
            str, contains title of overall plot
        """
        if len(param_list) == 0:
            return
        for param in param_list:
            if not isinstance(param, ImageParameter):
                try:
                    if not isinstance(self[param], ImageParameter):
                        raise SolverError(
                            f"{param} does not reference an ImageParameter."
                        )
                except KeyError as err:
                    raise SolverError from err
        n_rows = (1 + len(param_list)) // 2
        fig, _ = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, 4, fig)
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        for i in range(n_rows - 1 if len(param_list) % 2 else n_rows):
            ax_left = plt.subplot(gs[i : i + 1, :2])
            ax_right = plt.subplot(gs[i : i + 1, 2:])
            param = cast(
                ImageParameter,
                param_list[2 * i]
                if isinstance(param_list[2 * i], ImageParameter)
                else self[cast(Enum, param_list[2 * i])],
            )
            param.display(ax_left)
            param = cast(
                ImageParameter,
                param_list[2 * i + 1]
                if isinstance(param_list[2 * i + 1], ImageParameter)
                else self[cast(Enum, param_list[2 * i + 1])],
            )
            param.display(ax_right)
        if len(param_list) % 2:
            ax_center = plt.subplot(gs[-1, 1:3])
            param = cast(
                ImageParameter,
                param_list[-1]
                if isinstance(param_list[-1], ImageParameter)
                else self[param_list[-1]],
            )
            cast(ImageParameter, param).display(ax_center)
        fig.suptitle(title)
        plt.show()

    def dot_graph(
        self,
        node_args_func: Callable[[Parameter], Dict[str, str]] = dot_shape_func,
    ) -> Digraph:
        """Return a dot graph representation of the solver model."""
        vertex_parent_dict = self._get_dep_graph().vertex_parent_dict
        return vertex_parent_dict_to_dot(
            cast(Dict[Parameter, Set[Parameter]], vertex_parent_dict), node_args_func
        )


def vertex_parent_dict_to_dot(
    vertex_parent_dict: Dict[Parameter, Set[Parameter]],
    node_args_func: Optional[Callable[[Parameter], Dict[str, str]]] = None,
    invert=False,
) -> Digraph:
    """
    Convert a directed graph to a dot graph.

    Parameters
    ----------
    vertex_parent_dict : dict[Hashable, set[Hashable]]
        A dictionary representation of the directed graph
    node_args_func : Callable[[Hashable], dict[str, str]]
        A function handle to generate keyword arguments for the node of the dot graph
        depending on the current vertex
    """
    node_args_func = (lambda x: {}) if node_args_func is None else node_args_func
    dot = Digraph(comment="dependencies")
    for parameter in vertex_parent_dict:
        dot.node(parameter.enum_id.name, **node_args_func(parameter))
    for node, deps in vertex_parent_dict.items():
        for dep in deps:
            if invert:
                dot.edge(node.enum_id.name, dep.enum_id.name)
            else:
                dot.edge(dep.enum_id.name, node.enum_id.name)
    return dot
