# This file contains utilities to compute statistics and run algorithms on graphs

from pathlib import Path
from typing import Union, Optional
import csv
import networkx

class Graph:

    adjacents: dict[int, dict[int,float]]
    is_directed: bool
    _include_self_loops: bool
    nodes: set[int]

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
        self.nodes = set()

        def _add_interaction(_from, _to):
            if _from not in self.adjacents:
                self.adjacents[_from] = dict()
            if _to not in self.adjacents[_from]:
                self.adjacents[_from][_to] = 0
            self.adjacents[_from][_to] += 1
            self.nodes.add(_from)
            self.nodes.add(_to)

        with open(file, "r") as file_in:
            reader = csv.reader(file_in, delimiter=" ")
            current_row = 0
            for row in reader:
                from_user, to_user, timestamp = map(int, row)

                if from_timestamp <= timestamp <= to_timestamp and (include_self_loops or from_user != to_user):
                    # This if is to store all edges (up to a point), but only if no new nodes are added
                    if current_row < 2000 or (from_user in self.nodes and to_user in self.nodes):
                        _add_interaction(from_user, to_user)
                        if not make_directed and from_user != to_user:  # Add the reciprocal edge, if it's not a self-loop
                            _add_interaction(to_user, from_user)

                current_row += 1
                if log_frequency and not (current_row % log_frequency):
                    print(f"{current_row} rows loaded...")
                # Putting this here to make graphs in functionalities 2-4 easier to visualize
                # Remove this if to work on the whole graph
                if current_row == 500000:
                    break

        # Edges are distances, so weights should be inverse to the link's strength. Let's use reciprocals of interaction count
        for _from in self.adjacents:
            for _to in self.adjacents[_from]:
                self.adjacents[_from][_to] = 1.0/self.adjacents[_from][_to]


    def save_with_weights(self, file: Union[Path, str]) -> None:
        """
        Save the graph to a file, keeping pre-computed weights

        :param file: the file to which to save the graph
        """
        with open(file, "w") as file_out:
            writer = csv.writer(file_out, delimiter=" ")
            writer.writerow([
                1 if self.is_directed else 0,
                1 if self._include_self_loops else 0
            ])
            for _from in self.adjacents:
                for _to in self.adjacents[_from]:
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

    def to_networkx(self) -> networkx.Graph:
        """
        Convert the graph to NetworkX for drawing

        :return: the graph as a networkx.Graph
        """
        g = networkx.Graph() if not self.is_directed else networkx.DiGraph()

        for node_from, node_adjacents in self.adjacents.items():
            for node_to, weight in node_adjacents.items():
                if node_from not in g:
                    g.add_node(node_from)
                if node_to not in g:
                    g.add_node(node_to)
                g.add_weighted_edges_from([(node_from, node_to, weight)])

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


    def shortest_route(self, _from, _to) -> Optional[list[int]]:
        """
        Returns the shortest route between two nodes.

        :param _from: Starting node
        :param _to: End node
        :return: List of nodes making up the shortest path between the two nodes. It returns None if no path exists
        """
        from queue import PriorityQueue

        q = PriorityQueue()
        q.put((0, _from))  # The queue will contain (total_distance_from_`_from`, node_id) tuples,

        parents = dict()
        visited = set()
        distances = {_from: 0}
        while not q.empty():
            min_dist, current_node = q.get()
            visited.add(current_node)

            if current_node == _to:
                break

            for adjacent in self.adjacents.get(current_node, dict()):
                if adjacent in visited:
                    continue

                adjacent_distance = distances[current_node] + self.adjacents[current_node][adjacent]
                if not adjacent in distances or adjacent_distance < distances[adjacent]:
                    distances[adjacent] = adjacent_distance
                    parents[adjacent] = current_node
                    q.put((adjacent_distance, adjacent))

        if _to not in visited:
            return None

        path = [_to]
        current_node = _to
        while current_node != _from:
            current_node = parents[current_node]
            path.append(current_node)
        return list(reversed(path))


    def simple_path(self, source: int, target: int) -> Optional[list[int]]:
        """
        Given two nodes, returns a path between the two (with no necessary features, such as being the shortest).

        :param source: The source node
        :param target: The target node
        :return: A path between the two nodes, or None if one is not found
        """
        from queue import Queue

        queue = Queue()
        parents = {source: source}
        visited = {source}
        while not queue.empty():
            node = queue.get()

            for adjacent in self.adjacents[node]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    parents[adjacent] = node
                    queue.put(adjacent)

        if target not in visited:
            return None

        path = [target]
        current_node = target
        while current_node != source:
            current_node = parents[current_node]
            path.append(current_node)

        return list(reversed(path))


    def edmond_karp_min_cut(self, source: int, target: int) -> list[tuple[int, int]]:
        """
        Return the min-cut from source to target using the Edmond-Karp algorithm, modified to solve the problem
        in functionality 4.

        For simplicity, note that the graph is modified with its reduced version as the algorithm runs, and at the end
        modified edges are restored.

        :param source: The source node
        :param target: The target node
        :return: The list of edges making up the min-cut
        """
        from copy import deepcopy

        adj_copy = deepcopy(self.adjacents)

        min_cut = []
        while True:
            # path = self.simple_path(source, target)
            path = self.shortest_route(source, target)
            if not path:
                self.adjacents = adj_copy
                return min_cut

            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            min_edge_weight = min([self.adjacents[_from][_to] for _from, _to in edges])
            min_edge = [(_from, _to) for _from, _to in edges if self.adjacents[_from][_to] == min_edge_weight][0]

            del self.adjacents[min_edge[0]][min_edge[1]]

            min_cut.append(min_edge)

    def node_degree_centrality(self, node: int) -> int:
        """
        Return the node degree centrality. For directed graphs, it's defined as the sum
        of in-degree and out-degree.

        :param node: the node to compute centrality for.
        :return: The node's degree centrality
        """
        out_degree = len(self.adjacents[node])
        if not self.is_directed:
            return out_degree

        in_degree = len([k for k,v in self.adjacents.items() if node in v])
        return out_degree + in_degree

    def node_betweenness(self, node: int) -> float:
        """
        Return the node's betweenness, defined as the ratio of shortest paths across all node pairs
        that pass through the given node.

        :param node: The node to compute betweenness for
        :return: The node's betweenness
        """
        count = 0
        for node_i in self.node_ids:
            for node_j in self.node_ids:
                if node in [node_i, node_j]:
                    continue

                shortest_path = self.shortest_route(node_i, node_j)
                if node in shortest_path:
                    count += 1

        return count/(len(self.node_ids)*len(self.node_ids)-1)

    def node_closeness(self, node: int):
        """
        Return the node's closeness, defined as the reciprocal of the sum of distances from all other nodes

        :param node: The node to compute closeness for
        :return: The node's closeness
        """
        total_distance = 0
        for source_node in self.node_ids:
            if node == source_node:
                continue

            distance = len(self.shortest_route(source_node, node))-2 # -2 to remove source and target
            total_distance += distance

        return 1.0/total_distance

    def pagerank(self, num_iterations: int = 100) -> dict[int, float]:
        """
        Returns the PageRank values for all nodes. The algorithm is run for a predefined number of iterations.

        :param num_iterations: Number of items iterations
        :return: PageRank for all nodes in the graph
        """
        nodes = self.node_ids
        num_nodes = len(nodes)
        pr = {node: 1/num_nodes for node in nodes}
        damping = 0.85

        # Pre-compute in-links for performance
        in_links: dict[int, set] = {node: set() for node in self.nodes}
        for node, adjacents in self.adjacents.items():
            in_links[node].update(adjacents)

        for i in range(num_iterations):
            pr_next = dict()
            for node in nodes:
                pr_next[node] = ((1-damping)/num_nodes) + damping*sum([pr[in_link]/len(self.adjacents[in_link]) for in_link in in_links[node]])
            pr = pr_next

        return pr


def example_graph() -> Graph:
    """
    Return a simple, predefined graph to provide examples of visualizations

    :return: A simple, fully-connected graph
    """
    g = Graph(file=None)
    g.nodes = set(range(10))
    g.adjacents = {i: dict() for i in range(10)}
    for i in range(10):
        for j in range(10):
            if i != j:
                g.adjacents[i][j] = (i-j)**2
    g.is_directed = True
    g._include_self_loops = True
    return g


def random_graph(num_nodes: int, edge_probability: float) -> Graph:
    """
    Returns a random graph,

    :return:
    """
    from random import uniform

    g = Graph(file=None)
    g.nodes = set(range(num_nodes))
    g.adjacents = {i: dict() for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and uniform(0, 1) < edge_probability:
                g.adjacents[i][j] = 1
    g.is_directed = True
    g._include_self_loops = True
    return g


if __name__ == "__main__":
    from constants import MERGED_GRAPH_FILE

    graph = Graph(MERGED_GRAPH_FILE, True, True, log_frequency=1000000)
    print(f"# Nodes: {graph.num_nodes}")
    print(f"# Edges: {graph.num_edges}")
    node_ids = graph.node_ids
    print(f"Node ID Range: [{min(node_ids)}: {max(node_ids)}]")