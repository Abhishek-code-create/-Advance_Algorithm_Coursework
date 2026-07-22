"""
visualize.py
------------
Task 2 (c): Visualisation of algorithm execution.

Draws:
    1. dijkstra_tree.png - final shortest-path tree from Dijkstra, with
       each node labelled by its shortest distance from the source.
    2. prim_mst.png       - final Minimum Spanning Tree from Prim's
       algorithm, with edges in the order they were added shown by
       a colour gradient (early = dark, late = light).

Uses networkx purely for layout + drawing (not for the algorithms
themselves, which are implemented from scratch in dijkstra.py / prim.py).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

from graph import generate_random_graph
from dijkstra import dijkstra
from prim import prim_mst


def draw_dijkstra_tree(graph, source, filename="dijkstra_tree.png"):
    dist, prev, _ = dijkstra(graph, source)

    G = nx.DiGraph()
    for u, v, w in graph.edges():
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    # Highlight shortest-path tree edges
    tree_edges = [(prev[v], v) for v in graph.vertices if prev[v] is not None]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color="#cfe2ff", node_size=600)
    nx.draw_networkx_edges(G, pos, edge_color="#dddddd", arrows=True,
                            connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=tree_edges, edge_color="#d62728",
                            width=2.5, arrows=True, connectionstyle="arc3,rad=0.05")

    labels = {v: f"{v}\n({dist[v]})" for v in graph.vertices}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(f"Dijkstra Shortest-Path Tree from {source}\n(red edges = shortest-path tree, labels = distance from source)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def draw_prim_mst(graph, filename="prim_mst.png"):
    mst_edges, total_weight, steps = prim_mst(graph, track_steps=True)

    undirected = graph.to_undirected()
    G = nx.Graph()
    for u, v, w in undirected.edges():
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color="#cfe2ff", node_size=600)
    nx.draw_networkx_edges(G, pos, edge_color="#dddddd")

    # Colour MST edges by the order they were added (dark=early, light=late)
    n_steps = len(mst_edges)
    cmap = matplotlib.colormaps["viridis"]
    for i, (u, v, w) in enumerate(mst_edges):
        color = cmap(i / max(n_steps - 1, 1))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color], width=3)

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Prim's MST (total weight={total_weight})\n"
              f"edge colour: dark=added early -> light=added late")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def run():
    g = generate_random_graph(15, edge_density=0.2, seed=7)
    source = "C0"
    draw_dijkstra_tree(g, source)
    draw_prim_mst(g)


if __name__ == "__main__":
    run()
