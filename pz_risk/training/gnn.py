import torch
import torch.nn as nn
import torch.nn.functional as F

from training.egat_conv import EGATConv
from torch_geometric.nn import Linear


class GNN(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden):
        super().__init__()
        self.conv1 = EGATConv(-1, hidden, edge_dim=edge_feat, add_self_loops=False, fill_value='mean')
        self.conv2 = EGATConv(-1, hidden, edge_dim=hidden, add_self_loops=False, fill_value='mean')
        self.conv3 = EGATConv(-1, hidden, edge_dim=hidden, add_self_loops=False, fill_value='mean')

        self.classifier = Linear(-1, 1)
        # self.classifier1 = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # node_index_0 = [a - 2 for a in data.ptr.cpu().numpy()[1:]]
        # node_index_1 = [a - 1 for a in data.ptr.cpu().numpy()[1:]]
        #         if ret_att:
        x, a1, e = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.relu(x)
        e = e.squeeze()
        x = F.dropout(x, training=self.training)
        x, a2, e = self.conv2(x, edge_index, e, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        e = e.squeeze()
        x, a3, e = self.conv3(x, edge_index, e, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #         else:
        #             x = self.conv1(x, edge_index)
        #             x = F.relu(x)
        #             x = F.dropout(x, training=self.training)
        #             x = self.conv2(x, edge_index)
        #             x = F.relu(x)
        #             x = F.dropout(x, training=self.training)
        #             x = self.conv3(x, edge_index)
        #             x = F.relu(x)
        #             x = F.dropout(x, training=self.training)
        # print(x.shape)
        # x0 = x[node_index_0]
        # x1 = x[node_index_1]
        # print(x.shape)
        x = self.classifier(x)
        return x
