import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv  # Replaced GATConv with GCNConv
import time

# === Load the Cora dataset ===
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# === Define the GCN model ===
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        
        # First GCN layer
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# === Initialize the model, optimizer, and loss function ===
model = GCN(hidden_channels=8)  # 8 hidden channels
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# === Training function ===
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# === Test function for any set (train/val/test) ===
def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc

# === Open file to save results with UTF-8 encoding ===
with open('ex3a.txt', 'w', encoding='utf-8') as f:
    best_val_acc = 0
    best_model_state = None

    start_time = time.time()  # Start the timer

    for epoch in range(1, 201):
        loss = train()
        val_acc = test(data.val_mask)
        test_acc = test(data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        f.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}\n')

    model.load_state_dict(best_model_state)
    final_test_acc = test(data.test_mask)

    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate execution time

    f.write(f'\nFinal Test Accuracy (Best Val Model): {final_test_acc:.4f}, Time: {execution_time:.2f} seconds\n')

print("Results have been saved to 'ex3a.txt'.")
