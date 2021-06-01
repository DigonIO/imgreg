"""
Convenience functions for directed acyclic graphs

Author: Fabian A. Preiss.
"""
import itertools
from typing import Dict, Hashable, Set


class DAGraph:
    r"""
    Collection of functions for directed acyclic graphs.

    This class implements functions for directed acyclic graphs [#f3]_ [#f4]_.
    The graph is stored in a dictionary, where the keys define the identifiers for
    the vertices and the value contains the set of parents. This implementation
    is motivated for handling generic problems where dependencies appear (see
    dependency graphs [#f1]_).

    This module is for experimental use only, more extensive graph libraries for
    python are available under [#f2]_ [#f7]_ [#f8]_ [#f9]_.

    Parameters
    ----------
    vertex_parent_dict : dict[str, set[Hashable]]
        A dictionary mapping each vertex to all of its parents.
        It is assumed, that the input is a directed acyclic graph (connected
        or disconnected). Cycle Detection is not performed on the input.
    invert : boolean
        invert the edges of the input graph

    Notes
    -----
    A directed graph is an ordered pair :math:`G=\left(V,E\right)`, where

    * :math:`V` is a set of vertices (also nodes or points)
    * :math:`E\subseteq\left\{ (x,y)\mid(x,y)\in V^{2}\;\textrm{ and }\;x\neq y\right\}`
      is a set of ordered pairs of vertices called edges.

    To represent a graph, DAGraph takes a single python dictionary as input, where
    :math:`\mathtt{key},\,parent\in V` and

    .. math::
        \mathtt{value}=pa\left(\mathtt{key}\right)=\left\{ \left\{ parent\right\} |
        \left(\mathtt{key},parent\right)\in E\right\}

    References
    ----------
    .. [#f3] `Wikipedia, "Graph theory" <https://en.wikipedia.org/wiki/Graph_theory>`_
    .. [#f4] `Wikipedia,
             "Directed acyclic graph" <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_
    .. [#f1] `Wikipedia, "Dependency graph" <https://en.wikipedia.org/wiki/Dependency_graph>`_
    .. [#f6] `Wikipedia, "Tree (graph theory)" <https://en.wikipedia.org/wiki/Tree_(graph_theory)>`_
    .. [#f2] https://networkx.org/
    .. [#f7] https://graph-tool.skewed.de/
    .. [#f8] https://pygsp.readthedocs.io/en/stable/
    .. [#f9] https://igraph.org/python/
    """

    def __init__(self, vertex_parent_dict: Dict[Hashable, Set[Hashable]], invert=False):
        #  NOTE: Design decision: Use a dictionary to store all ascendants
        #  for each vertex in a set
        #  Contra:
        #    - one time computational cost with constructor
        #    - additional storage requirements
        #  Pro:
        #    - simplifies implementation for certain algorithms and
        #      allows faster access of often demanded properties
        self.invert = invert
        if invert:
            tmp_dag = DAGraph(vertex_parent_dict)
            self.__vertex_parent_dict = {
                key: tmp_dag.children(key) for key in tmp_dag.vertex_parent_dict
            }
            self.__vertex_ascendants_dict = {
                key: tmp_dag.descendants(key) for key in tmp_dag.vertex_parent_dict
            }
        else:
            self.__vertex_parent_dict = vertex_parent_dict
            self.__vertex_ascendants_dict = {
                key: self.__ascendants(key) for key in vertex_parent_dict.keys()
            }

    @property
    def vertex_parent_dict(self) -> Dict[Hashable, Set[Hashable]]:
        """
        Get a dictionary mapping each vertex to all of its parents.

        Returns
        -------
        dict
            vertex : Set of vertices
        """
        return self.__vertex_parent_dict

    @property
    def vertex_ascendants_dict(self) -> Dict[Hashable, Set[Hashable]]:
        """
        Get a dictionary mapping each vertex to all of its ascendants [#f6]_.

        Returns
        -------
        dict
            vertex : Set of vertices
        """
        return self.__vertex_ascendants_dict

    def __r_ascendants(
        self,
        vertex: Hashable,
        tested: Set[Hashable] = None,
        collected: Set[Hashable] = None,
    ):
        """
        Recursively ascend the graph and collect all ascendants of *vertex*.

        The final set contains all vertices from which *vertex* is
        reachable[#f5].
        Recursion overwrites *tested* and *collected*.

        Parameters
        ----------
        vertex : Hashable
            a vertex of the graph
        tested : Set
            Set of vertices tested for ascendants
        collected : Set
            Set of vertices that have been tagged as ascendants so far
        """
        vertex_parent_dict = self.__vertex_parent_dict
        tested = set() if tested is None else tested
        tested.update({vertex})
        collected = set() if collected is None else collected
        collected.update(vertex_parent_dict[vertex])
        for dep in vertex_parent_dict[vertex]:
            if dep not in tested and dep is not None:
                self.__r_ascendants(dep, tested=tested, collected=collected)

    def __ascendants(self, vertex: Hashable) -> Set[Hashable]:
        """
        Given *vertex* returns the set of all vertices from which *vertex* is reachable[#f5].

        .. [#f5] https://en.wikipedia.org/wiki/Reachability

        Parameters
        ----------
        vertex : Hashable
            a vertex of the graph

        Returns
        -------
        Set
            Set of vertices that are ascendants from vertex
        """
        res: Set[Hashable] = set()
        self.__r_ascendants(vertex, collected=res)
        return res

    def parents(self, vertex: Hashable, order=1) -> Set[Hashable]:
        """
        Given vertex, return a set containing all its parents.

        Returns
        -------
        Set
            Set of *parent*
        """
        current_parents = self.__vertex_parent_dict[vertex]
        if order > 1:
            return set(
                itertools.chain(
                    *(self.parents(parent, order - 1) for parent in current_parents)
                )
            )
        return current_parents

    def ascendants(self, vertex: Hashable) -> Set[Hashable]:
        """
        Given vertex, return a set containing all its ascendants [#f6]_.

        Returns
        -------
        Set
            Set of *ascendants*
        """
        return self.__vertex_ascendants_dict[vertex]

    def children(self, vertex: Hashable, order: int = 1) -> Set[Hashable]:
        """
        Given vertex, return a set containing all its children.

        Returns
        -------
        Set
            Set of *children*
        """
        current_children = {
            key
            for key in self.__vertex_parent_dict
            if vertex in self.__vertex_parent_dict[key]
        }
        if order > 1:
            return set(
                itertools.chain(
                    *(self.children(child, order - 1) for child in current_children)
                )
            )
        return current_children

    def descendants(self, vertex: Hashable) -> Set[Hashable]:
        """
        Given vertex, return a set containing all descendants [#f6]_.

        Returns
        -------
        Set
            Set of *descendants*
        """
        return {
            key
            for key in self.__vertex_ascendants_dict
            if vertex in self.__vertex_ascendants_dict[key]
        }
