# This file contains utilities to compute statistics and run algorithms on graphs

from pathlib import Path
from typing import Union, Optional
import csv

class Graph:

    adjacents: dict[int, dict[int,float]]
    is_directed: bool
    _include_self_loops: bool

    def __init__(
            self,
            file: Optional[Union[Path, str]],
            include_self_loops: bool = False,
            make_directed: bool = True,
            from_timestamp: int = 0,
            to_timestamp: int = 100000000000,
            log_frequency = 0,
    ):
        """
        Create a graph from a text file.

        The graph is always directed and weighted, and the weight of the edge from user i to user j is computed as
        the number of interactions between from the first to the latter (if directed), or between the two (if not).

        :param file: The file to load. If None, inits an empty graph
        :param include_self_loops: Whether to include self-loops (edges from a node to itself) in the graph.
        :param make_directed: Whether interactions are considered reciprocal.
        :param from_timestamp: Timestamp under which links are discarded.
        :param to_timestamp: Timestamp over which links are discarded.
        :param log_frequency: Print a log message each `log_frequency` lines. 0 to disable
        """
        if not file:
            return

        self.is_directed = make_directed
        self.adjacents = dict()
        self._include_self_loops = include_self_loops

        def _add_interaction(_from, _to):
            if _from not in self.adjacents:
                self.adjacents[_from] = dict()
            if _to not in self.adjacents[_from]:
                self.adjacents[_from][_to] = 0
            self.adjacents[_from][_to] += 1

        with open(file, "r") as file_in:
            reader = csv.reader(file_in, delimiter=" ")
            current_row = 0
            for row in reader:
                from_user, to_user, timestamp = map(int, row)
                if from_timestamp <= timestamp <= to_timestamp and (include_self_loops or from_user != to_user):
                    _add_interaction(from_user, to_user)
                    if not make_directed and from_user != to_user:  # Add the reciprocal edge, if it's not a self-loop
                        _add_interaction(to_user, from_user)
                current_row += 1
                if log_frequency and not (current_row % log_frequency):
                    print(f"{current_row} rows loaded...")

        # Edges are distances, so weights should be inverse to the link's strength. Let's use reciprocals of interaction count
        for _from in self.adjacents:
            for _to in self.adjacents[_from]:
                self.adjacents[_from][_to] = 1.0/self.adjacents[_from][_to]


    def save_with_weights(self, file: Union[Path, str], nodes_to_keep: set[int]) -> None:
        """
        Save the graph to a file, keeping pre-computed weights

        :param file: the file to which to save the graph
        :param nodes_to_keep: IDs of the nodes to be kept
        """
        with open(file, "w") as file_out:
            writer = csv.writer(file_out, delimiter=" ")
            writer.writerow([
                1 if self.is_directed else 0,
                1 if self._include_self_loops else 0
            ])
            for _from in self.adjacents:
                for _to in self.adjacents[_from]:
                    if _from in nodes_to_keep and _to in nodes_to_keep:
                        writer.writerow([_from, _to, self.adjacents[_from][_to]])

    @staticmethod
    def load_with_weights(cls, file: Union[Path, str]) -> "Graph":
        """
        Load the graph from a file, with pre-computed weights

        :param file: the file where the graph is stored.
        :return:
        """
        g = Graph(file=None)
        with open(file, "r") as file_in:
            reader = csv.reader(file_in, delimiter=" ")
            current_row = 0
            for row in reader:
                if current_row == 0:
                    g.is_directed, g._include_self_loops = map(int, row)
                    current_row += 1
                    continue
                from_user, to_user, weight = map(int, row)
                if from_user not in g.adjacents:
                    g.adjacents[from_user] = dict()
                g.adjacents[from_user][to_user] = weight
                current_row += 1
        return g

    def __len__(self) -> int:
        """
        Return the size of the graph, defined as the number of nodes it contains.

        :return: The size of the graph.
        """
        return self.num_nodes

    @property
    def num_nodes(self) -> int:
        """
        Return the number of nodes in the graph.

        :return: the number of nodes in the graph.
        """
        return len(self.adjacents)

    @property
    def num_edges(self) -> int:
        """
        Return the number of edges in the graph.

        :return: The number of edges
        """
        unique_edges = set()
        for _from, node_adjacents in self.adjacents.items():
            for adjacent in node_adjacents:
                if self.is_directed:
                    unique_edges.add((_from, adjacent))
                else:
                    # Don't count both (a,b) and (b,a). We can't just sum all edges and divide by 2
                    # because of self-loops
                    unique_edges.add((_from, adjacent) if _from < adjacent else (adjacent, _from))

        return len(unique_edges)

    @property
    def node_ids(self) -> set[int]:
        """
        Return the identifiers of all nodes in the graph

        :return: the identifiers of all nodes in the graph
        """
        ids = set(self.adjacents.keys())
        for node_adjacents in self.adjacents.values():
            ids.update(node_adjacents.keys())
        return ids

    def density(self, _num_nodes: Optional[int] = None, _num_edges: Optional[int] = None) -> float:
        """
        Return the density degree for this graph.

        The number of nodes and edges can be passed if precomputed

        :param _num_nodes: Number of nodes, if precomputed
        :param _num_edges: Number of edges, if precomputed
        :return:
        """
        num_nodes, num_edges = self.num_nodes if not _num_nodes else _num_nodes, self.num_edges if not _num_edges else _num_edges
        d = num_edges/num_nodes
        d /= (num_nodes-1 if not self._include_self_loops else num_nodes)  # Number of possible edges is higher if self-loops are possible
        if not self.is_directed:
            d *= 2
        return d


if __name__ == "__main__":
    from constants import MERGED_GRAPH_FILE

    graph = Graph(MERGED_GRAPH_FILE, True, True, log_frequency=1000000)
    print(f"# Nodes: {graph.num_nodes}")
    print(f"# Edges: {graph.num_edges}")
    node_ids = graph.node_ids
    print(f"Node ID Range: [{min(node_ids)}: {max(node_ids)}]")