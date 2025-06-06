import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network
import os
import time
import psutil

# Define Improved GAT model
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=0.6)
        self.conv4 = GATConv(hidden_dim * 8, num_classes, heads=1, dropout=0.6)
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return x

# Helper functions
def load_graph_from_file(file_path):
    edge_index = []
    num_nodes = 0
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'p':
                num_nodes = int(parts[2])
            elif parts[0] == 'e':
                edge_index.append([int(parts[1]) - 1, int(parts[2]) - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return num_nodes, edge_index

def accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    return correct / len(true_labels)

def track_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Graph visualization
def plot_colored_graph(graph, colors, instance_name, output_dir):
    num_nodes = len(graph.nodes)
    if len(colors) != num_nodes:
        print(f"Warning: Color array size {len(colors)} does not match the number of nodes {num_nodes}. Adjusting colors array.")
        colors = colors[:num_nodes]

    plt.figure(figsize=(7, 6))
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos,
            with_labels=False,
            node_color=colors,
            cmap=plt.cm.rainbow,
            node_size=100,
            edge_color='lightgray')

    unique_colors = sorted(set(colors))
    max_color = max(unique_colors) if unique_colors else 1

    if max_color == 0:
        patches = [mpatches.Patch(color=plt.cm.rainbow(0.0), label='Color 0')]
    else:
        patches = [mpatches.Patch(color=plt.cm.rainbow(i / max_color), label=f'Color {i}') for i in unique_colors]

    plt.legend(handles=patches, title="Clase GNN", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Colored Graph - {instance_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{instance_name}_colored_graph.png'))
    plt.savefig(os.path.join(output_dir, f'{instance_name}_colored_graph.svg'), format='svg')
    plt.close()
def interactive_graph(G, colors, instance_name, output_dir):
    net = Network(height='750px', width='100%', notebook=False)
    pos = nx.kamada_kawai_layout(G)
    cmap = plt.cm.rainbow

    # Ensure max_color is not zero
    max_color = max(colors) if max(colors) != 0 else 1

    # Add nodes first
    for i, node in enumerate(G.nodes()):
        node_id = str(node)  # Convert node ID to a string
        rgb = cmap(colors[i] / max_color)[:3]  # Safe division by max_color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        x, y = pos[node]
        net.add_node(node_id, label=str(node),
                     color=hex_color,
                     title=f"Node {node}, Color {colors[i]}",
                     x=float(x * 1000), y=float(y * 1000))

    # Then add edges after nodes are added
    for u, v in G.edges():
        net.add_edge(str(u), str(v))  # Ensure node IDs are strings

    net.set_options("""
    var options = {
      "physics": {
        "enabled": false
      }
    }
    """)
    net.save_graph(os.path.join(output_dir, f"{instance_name}_interactive.html"))


# Training setup
instances = [
    ("instances/queen5_5.col", 25, 5), 
    ("instances/queen6_6.col", 36, 7),
    ("instances/myciel5.col", 47, 6),
    ("instances/queen7_7.col", 49, 7),
    ("instances/queen8_8.col", 64, 9),
    ("instances/1-Insertions_4.col", 67, 4),
    ("instances/huck.col", 74, 11), 
    ("instances/jean.col", 80, 10),
    ("instances/queen9_9.col", 81, 10),
    ("instances/david.col", 87, 11),
    ("instances/mug88_1.col", 88, 4),
    ("instances/myciel6.col", 95, 7),
    ("instances/queen8_12.col", 96, 12),
    ("instances/games120.col", 120, 9), 
    ("instances/queen11_11.col", 121, 11),
    ("instances/anna.col", 138, 11),
    ("instances/2-Insertions_4.col", 149, 4),
    ("instances/queen13_13.col", 169, 13),
    ("instances/myciel7.col", 191, 8),
    ("instances/homer.col", 561, 13)
]

hidden_dim = 8
learning_rate = 0.005
num_epochs = 5300
output_dir = 'table1_results_32hd_4lay'
os.makedirs(output_dir, exist_ok=True)

table_1_results = []

# Training loop
for instance in instances:
    file_path, num_nodes, num_classes = instance

    model = GAT(num_node_features=3, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    _, edge_index = load_graph_from_file(file_path)
    x = torch.rand((num_nodes, 3))
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-9)
    data = Data(x=x, edge_index=edge_index)
    true_labels = torch.randint(0, num_classes, (num_nodes,))

    start_time = time.time()
    ram_usage = []
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, true_labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        ram_usage.append(track_ram_usage())
        epoch_times.append(time.time() - epoch_start_time)

    model.eval()
    out = model(data.x, data.edge_index)
    predicted_colors = torch.argmax(out, dim=1)
    predicted_chromatic_number = predicted_colors.max().item() + 1
    acc = accuracy(predicted_colors, true_labels)
    total_time = time.time() - start_time

    table_1_results.append({
        'Instance': os.path.basename(file_path),
        'Size': num_nodes,
        'True χ': num_classes,
        'Predicted χ (GNN)': predicted_chromatic_number,
        'Predicted Colors': predicted_colors.tolist(),
        'Execution Time (s)': total_time
    })

    instance_name = os.path.basename(file_path)

    # RAM usage plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epochs), ram_usage, label="RAM Usage (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("RAM Usage (MB)")
    plt.title(f"RAM Usage for {instance_name}")
    plt.savefig(os.path.join(output_dir, f"{instance_name}_ram_usage.png"))
    plt.close()

    # Cumulative time plot
    cumulative_times = [sum(epoch_times[:i+1]) for i in range(len(epoch_times))]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), cumulative_times, marker='o', linewidth=2, color='teal')
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Execution Time (s)")
    plt.title(f"Cumulative Execution Time - {instance_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{instance_name}_cumulative_execution_time.png"))
    plt.close()

    # Visualization
    G = nx.Graph()
    edge_list = edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    plot_colored_graph(G, predicted_colors.tolist(), instance_name, output_dir)
    interactive_graph(G, predicted_colors.tolist(), instance_name, output_dir)

# Save results
with open(os.path.join(output_dir, 'table_1_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Table 1: Results for Color02/03/04 instances:\n")
    for result in table_1_results:
        f.write(f"Instance: {result['Instance']}\n")
        f.write(f"Size: {result['Size']}\n")
        f.write(f"True χ: {result['True χ']}\n")
        f.write(f"Predicted χ (GNN): {result['Predicted χ (GNN)']}\n")
        f.write(f"Predicted Colors: {result['Predicted Colors']}\n")
        f.write(f"Execution Time (s): {result['Execution Time (s)']}\n\n")
