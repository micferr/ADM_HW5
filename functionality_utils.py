# This file contains utilities specific to the homework's 4 functionalities to be implemented, e.g. for plotting data
from typing import Optional

from IPython.display import display
import pandas as pd
import networkx

from graph_utils import Graph


def functionality_1():
    """
    Implements the frontend for the first functionality
    """
    graph_file = input("Input the path to the graph file...")

    graph = Graph(graph_file)

    is_directed = graph.is_directed
    num_users = graph.num_nodes
    num_links = graph.num_edges  # Note that this is _not_ the number of comments/answers
    average_links_per_user = num_links/num_users
    density = graph.density(num_users, num_links)
    is_sparse = True  # Empirically these graphs are always sparse, otherwise we could set a threshold for density

    df = pd.DataFrame.from_dict({
        "Is Directed": [is_directed],
        "# Users": [num_users],
        "# Links": [num_links],
        "Avg. Links per User": [average_links_per_user],
        "Density": [density],
        "Is Sparse": [is_sparse]
    }, orient="index", columns=["Value"])

    display(df)

    def node_density(node):
        d = len(graph.adjacents.get(node, {}))  # Out-degree
        d += len([1 for n in graph.nodes if node in graph.adjacents.get(n, {})])  # In-degree
        return d

    densities = [node_density(node) for node in graph.nodes]
    pd.DataFrame(densities).plot(kind='density')


def functionality_2(on_graph: Optional[Graph] = None):
    """
    Implements the frontend for the second functionality.

    :param on_graph: if provided, run the method on the argument file without requesting the user to input it.
    """
    if not on_graph:
        graph_file = input("Input the path to the graph file...")
        print(f"Graph file: {graph_file}")
        min_timestamp = 1199142000 # Jan 1 2008, rough estimate since S/O was launched in 2008
        max_timestamp = 1640000000 # Falls on Dec 20 2021, right after the homework's deadline
        time_step = max_timestamp - min_timestamp
        # Run on the graph by splitting it by equal time intervals
        graphs = [
            Graph(
                graph_file,
                from_timestamp=min_timestamp+time_step//4*i,
                to_timestamp=min_timestamp+time_step//4*(i+1)
            )
            for i in range(4)
        ]
    else:
        # Run on a single graph
        graphs = [on_graph]

    available_metrics = ["Betweenness", "Centrality", "Closeness", "PageRank"]
    metric = input("Insert a metric [Betweenness, Centrality, Closeness, PageRank]...")

    if metric not in available_metrics:
        print("Metric not valid")
        return
    print(f"Chosen metric: {metric}")

    node = int(input("Input a node to compute the metric on..."))
    print(f"Chosen node: {node}")

    for graph in graphs:
        if metric == "Betweenness":
            metric_value = graph.node_betweenness(node)
        elif metric == "Centrality":
            metric_value = graph.node_degree_centrality(node)
        elif metric == "Closeness":
            metric_value = graph.node_closeness(node)
        else:
            metric_value = graph.pagerank()[node]

        nx_graph = graph.to_networkx()

        # Keep only the relevant nodes to plot them
        nodes_to_discard = [
            n
            for n in nx_graph.nodes
            if not nx_graph.has_edge(node, n) and not nx_graph.has_edge(n, node)
        ]
        nx_graph.remove_nodes_from(nodes_to_discard)

        node_colors = ["yellow" if n == node else "blue" for n in nx_graph.nodes]
        networkx.draw_circular(nx_graph, node_color=node_colors)

        print(f"{metric}: {metric_value}")


def functionality_3(on_graph: Optional[Graph] = None):
    """
    Implements the frontend for the third functionality.

    :param on_graph: if provided, run the method on the argument file without requesting the user to input it.
    """
    if not on_graph:
        graph_file = input("Input the path to the graph file...")
        print(f"Graph file: {graph_file}")
        graph = Graph(graph_file)
    else:
        graph = on_graph

    nodes = list(map(int, input("Input the space-separated list of nodes to visit (including first and last)...").split(" ")))
    print(f"Nodes to traverse: {nodes}")

    # Check that all nodes are in the graph
    all_nodes = graph.node_ids
    for node in nodes:
        if node not in all_nodes:
            print(f"{node} is not a node in the graph.")
            return

    paths = []
    for i in range(len(nodes)-1):
        _from, _to = nodes[i], nodes[i+1]
        path = graph.shortest_route(_from, _to)

        if path is None:
            print("Not possible")
            return

        paths.append(path)

    # Merge all paths in a single walk
    walk = paths[0]
    for path in paths[1:]:
        # Remove the starting node from each subsequent path, as it is already contained in the previous one
        walk.extend(path[1:])

    nx_graph = graph.to_networkx()
    edges = set((walk[i], walk[i+1]) for i in range(len(walk)-1))  # to use for coloring
    edge_colors = ["red" if (u,v) in edges else "black" for u,v in nx_graph.edges()]
    node_colors = ["yellow" if node in nodes else "blue" for node in nx_graph.nodes]

    print(f"Walk: {walk}")

    networkx.draw_circular(nx_graph, edge_color=edge_colors, node_size=3, node_color=node_colors)


def functionality_4(on_graph: Optional[Graph] = None):
    """
    Implements the frontend for the third functionality.

    :param on_graph: if provided, run the method on the argument file without requesting the user to input it.
    """
    if not on_graph:
        graph_file = input("Input the path to the graph file...")
        print(f"Graph file: {graph_file}")
        graph = Graph(graph_file)
    else:
        graph = on_graph

    nodes = list(map(int, input("Input the two nodes to disconnect...").split(" ")))
    if len(nodes) != 2:
        print("Invalid number of nodes supplied.")
        return

    print(f"Source: {nodes[0]}, Target: {nodes[1]}")

    min_cut = graph.edmond_karp_min_cut(nodes[0], nodes[1])
    if not min_cut:
        print("The nodes are already disconnected.")
        return

    nx_graph = graph.to_networkx()
    edge_colors = ["red" if (u, v) in min_cut else "black" for u, v in nx_graph.edges()]
    node_colors = ["yellow" if node in nodes else "blue" for node in nx_graph.nodes]

    print(f"Min cut: {min_cut}")

    networkx.draw_circular(nx_graph, edge_color=edge_colors, node_size=3, node_color=node_colors)