import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
import psutil

# ===================== MODEL DEFINITIONS =====================
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * 4, num_classes, heads=1, dropout=0.6)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

# ===================== ACCURACY FUNCTION =====================
def accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    return correct / len(true_labels)

# ===================== DISTRIBUTIONS FOR TABLE 2 =====================
synthetic_distributions = {
    "Power Law Tree": lambda: nx.random_powerlaw_tree(n=50, tries=1000, seed=42),
    "Small-world": lambda: nx.watts_strogatz_graph(n=50, k=4, p=0.1, seed=42),
    "Holme and Kim": lambda: nx.powerlaw_cluster_graph(n=50, m=2, p=0.3, seed=42),
}

# ===================== TRAINING SETTINGS =====================
hidden_dim = 64
learning_rate = 0.005
num_epochs = 5300
num_classes = 5  # assuming 5 colors (labels)

# ===================== MAIN LOOP =====================
table_2_results = {}
accuracy_per_epoch = {name: [] for name in synthetic_distributions.keys()}
execution_time_per_epoch = {name: [] for name in synthetic_distributions.keys()}
ram_usage_per_epoch = {name: [] for name in synthetic_distributions.keys()}
total_execution_time = {name: 0 for name in synthetic_distributions.keys()}  # Track total execution time

for name, generator in synthetic_distributions.items():
    print(f"Training model for: {name}")
    G = generator()
    G = nx.convert_node_labels_to_integers(G)
    data = from_networkx(G)

    num_nodes = G.number_of_nodes()
    edge_index = data.edge_index

    model = GAT(num_node_features=3, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Random normalized features
    x = torch.rand((num_nodes, 3))
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-9)
    data.x = x

    # Random labels for general evaluation
    true_labels = torch.randint(0, num_classes, (num_nodes,))

    start_time_training = time.time()  # Start time for entire training process

    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for epoch
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, true_labels)
        loss.backward()
        optimizer.step()

        # Record execution time
        execution_time = time.time() - start_time
        execution_time_per_epoch[name].append(execution_time)

        # Record RAM usage
        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        ram_usage_per_epoch[name].append(ram_usage)

        # Accuracy calculation at each epoch
        model.eval()
        with torch.no_grad():
            out_val = model(data.x, data.edge_index)
            preds_val = torch.argmax(out_val, dim=1)
            acc_val = accuracy(preds_val, true_labels)

        accuracy_per_epoch[name].append(acc_val)

    # Final testing
    model.eval()
    out = model(data.x, data.edge_index)
    predicted = torch.argmax(out, dim=1)
    acc = accuracy(predicted, true_labels)

    # Track the total execution time for the training process
    total_execution_time[name] = time.time() - start_time_training

    table_2_results[name] = {
        "accuracy": round(acc * 100, 2),
        "execution_time": total_execution_time[name]  # Store the total execution time
    }

# ===================== RESULTS =====================
print("\nTable 2 - GNN Results on Synthetic Graphs:")
for dist, result in table_2_results.items():
    print(f"{dist}: {result['accuracy']:.2f}% accuracy, {result['execution_time']:.2f} seconds execution time")

# ===================== SAVE TO FILE =====================
# Ensure that the table2_results directory exists
if not os.path.exists('table2_results'):
    os.makedirs('table2_results')

# Save the results to a text file inside the table2_results directory
with open("table2_results/table_2_results.txt", "w", encoding="utf-8") as f:
    f.write("Table 2: GNN Results on Synthetic Distributions (compared to Lemos et al.)\n\n")
    for dist, result in table_2_results.items():
        f.write(f"{dist}: {result['accuracy']:.2f}% accuracy, {result['execution_time']:.2f} seconds execution time\n")

# ===================== GRAPHICAL REPRESENTATION OF ACCURACY EVOLUTION OVER EPOCHS =====================
plt.figure(figsize=(10, 6))
for name, accs in accuracy_per_epoch.items():
    plt.plot(range(1, num_epochs + 1), accs, label=name)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model GNN Accuracy Evolution Over Epochs for Synthetic Distributions', fontsize=14)
plt.legend(title='Graph Distributions', loc='lower right')
plt.ylim(0, 1)

# Saving the accuracy graph in the table2_results folder
plt.tight_layout()
plt.savefig("table2_results/accuracy_per_epoch_graph.png")

# ===================== GRAPHICAL REPRESENTATION OF CUMULATIVE EXECUTION TIME (PROGRESSIVE LINE) =====================
plt.figure(figsize=(10, 6))
for name, times in execution_time_per_epoch.items():
    cumulative_times = [sum(times[:i+1]) for i in range(len(times))]
    plt.plot(range(1, num_epochs + 1), cumulative_times, label=name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cumulative Execution Time (seconds)', fontsize=12)
plt.title('Cumulative Execution Time per Epoch for All Distributions', fontsize=14)
plt.legend(title='Graph Distributions', loc='upper left')
plt.tight_layout()
plt.savefig("table2_results/combined_cumulative_execution_time_graph.png")

# ===================== GRAPHICAL REPRESENTATION OF RAM USAGE =====================
plt.figure(figsize=(10, 6))
for name, usages in ram_usage_per_epoch.items():
    plt.plot(range(1, num_epochs + 1), usages, label=name)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RAM Usage (MB)', fontsize=12)
plt.title('Model GNN RAM Usage per Epoch', fontsize=14)
plt.legend(title='Graph Distributions', loc='upper right')
plt.tight_layout()
plt.savefig("table2_results/ram_usage_per_epoch_graph.png")

# ===================== SHOW ALL GRAPHS SIMULTANEOUSLY =====================
plt.show()
