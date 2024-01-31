from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GCN_Classifier(nn.Module):
    def __init__(self, num_features, hidden_layer, num_classes):
        super(GCN_Classifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x