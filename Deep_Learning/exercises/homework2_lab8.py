import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
import time

# === Load the Cora dataset ===
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # the actual graph

# === Define the GAT model with a variable number of layers ===
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers):
        super().__init__()
        torch.manual_seed(1234567)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels, heads=heads, dropout=0.6))
        
        for _ in range(num_layers - 2):  # Intermediate layers
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6))
        
        self.convs.append(GATConv(hidden_channels * heads, dataset.num_classes, heads=1, concat=False, dropout=0.6))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        # Last output layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# === Initialize the file to save results ===
with open('ex2.txt', 'w', encoding='utf-8') as f:
    # === Configurations to test ===
    hidden_channels_list = [8, 16, 32]  # Hidden feature dimensions
    layers_list = [2, 3, 4]  # Number of layers

    for hidden_channels in hidden_channels_list:
        for num_layers in layers_list:
            # Create model for each combination
            model = GAT(hidden_channels=hidden_channels, heads=8, num_layers=num_layers)
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

            # === Training loop + validation + saving the best model ===
            best_val_acc = 0
            best_model_state = None

            start_time = time.time()  # Start the timer

            for epoch in range(1, 201):
                loss = train()
                val_acc = test(data.val_mask)
                test_acc = test(data.test_mask)

                # Save the model if it has the best performance on the validation set
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

                # Save the results of each epoch in ex2.txt file
                f.write(f'Hidden Channels: {hidden_channels}, Layers: {num_layers}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}\n')

            # Load the best model and test on the test_mask
            model.load_state_dict(best_model_state)
            final_test_acc = test(data.test_mask)

            end_time = time.time()  # End the timer
            execution_time = end_time - start_time  # Calculate execution time

            f.write(f'\nFinal Test Accuracy (Best Val Model) - Hidden Channels: {hidden_channels}, Layers: {num_layers}: {final_test_acc:.4f}, Time: {execution_time:.2f} seconds\n')

print("Results have been saved to 'ex2.txt'.")
