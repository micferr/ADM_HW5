# This file contains utilities specific to the homework's 4 functionalities to be implemented, e.g. for plotting data

from IPython.display import display
from pathlib import Path
import pandas as pd

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

