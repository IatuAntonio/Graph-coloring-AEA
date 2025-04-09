import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import time
import psutil

# Define Improved GAT model
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1)
        self.conv3 = GATConv(hidden_dim * 4, num_classes, heads=1, dropout=0.1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
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

# Params
hidden_dim = 64
learning_rate = 0.001
num_epochs = 5300
table_1_results = []

os.makedirs('table1_results', exist_ok=True)

# Train on each instance
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
    epoch_times = []  # Track per-epoch execution time

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

    # RAM usage plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epochs), ram_usage, label="RAM Usage (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("RAM Usage (MB)")
    plt.title(f"RAM Usage for {os.path.basename(file_path)}")
    plt.savefig(f'table1_results/{os.path.basename(file_path)}_ram_usage.png')
    plt.close()

    # Progressive cumulative execution time plot
    cumulative_times = [sum(epoch_times[:i+1]) for i in range(len(epoch_times))]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), cumulative_times, marker='o', linewidth=2, color='teal')
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Execution Time (s)")
    plt.title(f"Cumulative Execution Time - {os.path.basename(file_path)}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'table1_results/{os.path.basename(file_path)}_cumulative_execution_time.png')
    plt.close()

# Save text results
with open('table1_results/table_1_results.txt', 'w', encoding='utf-8') as f:
    f.write("Table 1: Results for Color02/03/04 instances:\n")
    for result in table_1_results:
        f.write(f"Instance: {result['Instance']}\n")
        f.write(f"Size: {result['Size']}\n")
        f.write(f"True χ: {result['True χ']}\n")
        f.write(f"Predicted χ (GNN): {result['Predicted χ (GNN)']}\n")
        f.write(f"Predicted Colors: {result['Predicted Colors']}\n")
        f.write(f"Execution Time (s): {result['Execution Time (s)']}\n\n")

# Visualization functions (same as original)
def plot_colored_graph(graph, colors, instance_name):
    num_nodes = len(graph.nodes)
    if len(colors) != num_nodes:
        print(f"Warning: Color array size {len(colors)} does not match the number of nodes {num_nodes}. Adjusting colors array.")
        colors = colors[:num_nodes]
    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow, edge_color='gray', node_size=700)
    plt.title(f"Colored Graph - {instance_name}")
    plt.savefig(f'table1_results/{instance_name}_colored_graph.png')
    plt.close()

def visualize_embedding(h, colors, instance_name):
    h = h.detach().cpu().numpy()
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    h = (h - h.mean()) / (h.std() + 1e-9)
    if h.shape[1] > 1:
        h = TSNE(n_components=2, perplexity=min(2, h.shape[0] - 1)).fit_transform(h)
    else:
        h = torch.cat((torch.linspace(-1, 1, h.shape[0]).unsqueeze(1), torch.zeros(h.shape[0], 1)), dim=1).numpy()
    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(h[:, 0], h[:, 1], s=140, c=colors, cmap="Set2")
    plt.title(f"Embedding - {instance_name}")
    plt.legend(*scatter.legend_elements(), title="Node Colors")
    plt.savefig(f'table1_results/{instance_name}_embedding.png')
    plt.close()

# Visualize all
for instance, result in zip(instances, table_1_results):
    instance_name = result['Instance']
    predicted_colors = result['Predicted Colors']
    G = nx.Graph()
    _, edge_index = load_graph_from_file(f"instances/{instance_name}")
    edge_list = edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    plot_colored_graph(G, predicted_colors, instance_name)
    visualize_embedding(torch.tensor(predicted_colors), predicted_colors, instance_name)
