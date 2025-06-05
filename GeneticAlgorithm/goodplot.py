import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pyvis.network import Network

def read_col_file(filepath):
    graph = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('p'):
                _, _, num_nodes, _ = line.strip().split()
                num_nodes = int(num_nodes)
                graph = {i: [] for i in range(num_nodes)}
            elif line.startswith('e'):
                _, u, v = line.strip().split()
                u, v = int(u) - 1, int(v) - 1
                graph[u].append(v)
                graph[v].append(u)
    return graph

def draw_static(graph, colors, output_prefix):
    G = nx.Graph(graph)
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow,
            node_size=200, edge_color='gray')

    unique_colors = sorted(set(colors))
    patches = [mpatches.Patch(color=plt.cm.rainbow(i / max(unique_colors)), label=f'Color {i}')
               for i in unique_colors]
    plt.legend(handles=patches, title="Colors", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Colored Graph - {output_prefix}")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_graph.png")
    plt.savefig(f"{output_prefix}_graph.svg")
    plt.close()

def draw_interactive(graph, colors, output_prefix):
    G = nx.Graph(graph)
    pos = nx.kamada_kawai_layout(G)
    net = Network(height="700px", width="100%", notebook=False)
    cmap = plt.cm.rainbow
    max_color = max(colors)

    for i, node in enumerate(G.nodes()):
        rgb = cmap(colors[i] / max_color)[:3]
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        x, y = pos[node]
        net.add_node(str(node), label=str(node), color=hex_color, x=float(x*1000), y=float(y*1000),
                     title=f"Node {node}, Color {colors[i]}")

    for u, v in G.edges():
        net.add_edge(str(u), str(v))

    net.set_options("""var options = {"physics": {"enabled": false}}""")
    net.save_graph(f"{output_prefix}_interactive.html")

if __name__ == "__main__":
    # ======= MODIFICĂ ASTEA =========
    graph_name = "queen13_13"  # numele fișierului din folderul `instances`, fără extensie
    solution = [18, 5, 14, 9, 12, 7, 3, 13, 2, 4, 19, 20, 0, 17, 21, 1, 7, 11, 5, 4, 0, 10, 6, 2, 3, 15, 3, 11, 2, 8, 13, 0, 6, 5, 14, 1, 18, 21, 17, 6, 15, 10, 14, 19, 1, 17, 8, 22, 7, 16, 0, 11, 11, 1, 7, 2, 5, 6, 16, 19, 9, 0, 10, 22, 12, 4, 2, 8, 13, 16, 12, 18, 17, 1, 23, 14, 15, 5, 19, 18, 17, 0, 23, 21, 11, 2, 15, 3, 13, 6, 22, 23, 9, 3, 21, 4, 10, 7, 16, 0, 5, 8, 11, 2, 8, 13, 11, 12, 15, 22, 1, 9, 3, 10, 0, 14, 4, 2, 10, 5, 16, 8, 3, 15, 4, 19, 13, 22, 23, 6, 13, 8, 20, 4, 21, 19, 9, 3, 18, 12, 1, 16, 10, 7, 12, 23, 18, 6, 8, 0, 1, 17, 20, 5, 4, 3, 5, 14, 4, 1, 0, 15, 12, 6, 8, 2, 9, 18, 19]
    # ==================================

    path = os.path.join("./instances", f"{graph_name}.col")
    graph = read_col_file(path)

    output_prefix = f"./new_graphs/{graph_name}"
    os.makedirs("./new_graphs", exist_ok=True)

    draw_static(graph, solution, output_prefix)
    draw_interactive(graph, solution, output_prefix)

    print(f"Saved plots for {graph_name}.")
