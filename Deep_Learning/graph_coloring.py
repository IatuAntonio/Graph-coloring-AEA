import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define Improved GAT model
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=0.2)  # Add heads to extend learning capacity
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.2)  # Larger dimension
        self.conv3 = GATConv(hidden_dim * 4, num_classes, heads=1, dropout=0.2)  # Reduce to num_classes
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)  
        return x  

# Function to load .col files and build graphs
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

# Function to calculate accuracy
def accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    return correct / len(true_labels)

# Instances for testing
instances = [
    ("queen5_5.col", 25, 5), 
    ("queen6_6.col", 36, 7),
    ("myciel5.col", 47, 6),
    ("queen7_7.col", 49, 7),
    ("queen8_8.col", 64, 9),
    ("1-Insertions_4.col", 67, 4),
    ("huck.col", 74, 11), 
    ("jean.col", 80, 10),
    ("queen9_9.col", 81, 10),
    ("david.col", 87, 11),
    ("mug88_1.col", 88, 4),
    ("myciel6.col", 95, 7),
    ("queen8_12.col", 96, 12),
    ("games120.col", 120, 9), 
    ("queen11_11.col", 121, 11),
    ("anna.col", 138, 11),
    ("2-Insertions_4.col", 149, 4),
    ("queen13_13.col", 169, 13),
    ("myciel7.col", 191, 8),
    ("homer.col", 561, 13)
]

# Model parameters
hidden_dim = 64  # Larger hidden dimension to capture more complex relationships
learning_rate = 0.001
num_epochs = 300
table_1_results = []

# Loop through each instance
for instance in instances:
    file_path, num_nodes, num_classes = instance

    model = GAT(num_node_features=3, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Load the graph
    _, edge_index = load_graph_from_file(file_path)

    # Generate random features and labels
    x = torch.rand((num_nodes, 3))
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-9)  # Improved normalization
    data = Data(x=x, edge_index=edge_index)
    true_labels = torch.randint(0, num_classes, (num_nodes,))

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, true_labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)  

    # Test the model
    model.eval()
    out = model(data.x, data.edge_index)
    predicted_colors = torch.argmax(out, dim=1)
    predicted_chromatic_number = predicted_colors.max().item() + 1  
    acc = accuracy(predicted_colors, true_labels)

    # Save results for this instance
    table_1_results.append({
        'Instance': file_path,
        'Size': num_nodes,
        'True χ': num_classes,
        'Predicted χ (GNN)': predicted_chromatic_number,
        'Predicted Colors': predicted_colors.tolist()
    })

# Save all results to a file **before visualizations**
with open('table_1_results.txt', 'w', encoding='utf-8') as f:
    f.write("Table 1: Results for Color02/03/04 instances:\n")
    for result in table_1_results:
        f.write(f"Instance: {result['Instance']}\n")
        f.write(f"Size: {result['Size']}\n")
        f.write(f"True χ: {result['True χ']}\n")
        f.write(f"Predicted χ (GNN): {result['Predicted χ (GNN)']}\n")
        f.write(f"Predicted Colors: {result['Predicted Colors']}\n")
        f.write("\n")

# Functions for visualization
def plot_colored_graph(graph, colors, instance_name):
    # Ensure that the length of colors matches the number of nodes in the graph
    num_nodes = len(graph.nodes)
    if len(colors) != num_nodes:
        print(f"Warning: Color array size {len(colors)} does not match the number of nodes {num_nodes}. Adjusting colors array.")
        colors = colors[:num_nodes]  # Trim colors to match number of nodes

    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow, edge_color='gray', node_size=700)
    plt.title(f"Colored Graph - {instance_name}")
    plt.gcf().canvas.manager.set_window_title(f"Colored Graph - {instance_name}")
    plt.show(block=False)

def visualize_embedding(h, colors, instance_name):
    h = h.detach().cpu().numpy()

    # Check if `h` is 1D and reshape it to 2D if necessary
    if h.ndim == 1:
        h = h.reshape(-1, 1)

    # Normalization
    h = (h - h.mean()) / (h.std() + 1e-9)

    # Apply t-SNE only if we have more than 1 feature
    if h.shape[1] > 1:
        h = TSNE(n_components=2, perplexity=min(2, h.shape[0] - 1)).fit_transform(h)
    else:
        # Generate random X-Y coordinates if t-SNE cannot be used
        h = torch.cat((torch.linspace(-1, 1, h.shape[0]).unsqueeze(1), torch.zeros(h.shape[0], 1)), dim=1).numpy()

    # Plot
    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(h[:, 0], h[:, 1], s=140, c=colors, cmap="Set2")
    plt.title(f"Embedding - {instance_name}")
    plt.gcf().canvas.manager.set_window_title(f"Embedding - {instance_name}")
    plt.legend(*scatter.legend_elements(), title="Node Colors")
    plt.show(block=False)


# Visualize the graphs and embeddings **after saving the results**
for instance, result in zip(instances, table_1_results):
    file_path = result['Instance']
    predicted_colors = result['Predicted Colors']

    G = nx.Graph()
    _, edge_index = load_graph_from_file(file_path)
    edge_list = edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)

    # Ensure that the number of predicted colors matches the number of nodes
    plot_colored_graph(G, predicted_colors, file_path)
    visualize_embedding(torch.tensor(predicted_colors), predicted_colors, file_path)

    # Wait for windows to close before continuing
    while plt.fignum_exists(1) or plt.fignum_exists(2):
        plt.pause(0.1)
