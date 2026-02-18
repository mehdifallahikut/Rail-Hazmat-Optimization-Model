# graph_utils.py
import networkx as nx
import matplotlib.pyplot as plt

# graph_utils.py
import networkx as nx
import matplotlib.pyplot as plt


def to_bidirected(network_graph: nx.Graph) -> nx.DiGraph:
    G = nx.DiGraph()
    # ---------------------------------------------------------
    G.add_nodes_from(network_graph.nodes(data=True))

    for u, v, d in network_graph.edges(data=True):
        G.add_edge(u, v, **d)
        G.add_edge(v, u, **d)
    return G


def draw_network(network_graph: nx.Graph, key_nodes: list[str], figsize=(25, 10)) -> None:
    pos = nx.get_node_attributes(network_graph, 'pos')
    node_colors = ['skyblue' if n in key_nodes else 'lightgreen' for n in network_graph.nodes()]
    node_sizes = [400 if n in key_nodes else 200 for n in network_graph.nodes()]
    plt.figure(figsize=figsize)
    nx.draw(network_graph, pos, with_labels=True, node_color=node_colors,
            edge_color='gray', node_size=node_sizes, font_size=10, font_weight='bold')
    edge_labels = {(u, v): f"c:{d['cost']:.1f}\ne:{d['exposure']:.1f}"
                   for u, v, d in network_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Generated Rail Network")
    plt.show()
