import torch
import torch.nn as nn
import torch.nn.functional as F

from training.egat_conv import EGATConv
from training.eheat_conv import EHEATConv
from training.ehat_conv import EHATConv
from torch_geometric.nn import Linear, HEATConv, HeteroLinear


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


class HetroGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden):
        super().__init__()
        self.conv1 = EHEATConv(node_feat, hidden, 3, 4, 4, edge_feat, hidden)
        self.conv2 = EHEATConv(hidden, hidden, 3, 4, 4, hidden, hidden)
        self.conv3 = EHEATConv(hidden, hidden, 3, 4, 4, hidden, hidden)

        self.node_classifier = HeteroLinear(hidden, 1, 3)
        self.edge_classifier = HeteroLinear(hidden, 1, 4)
        # self.classifier1 = nn.Linear(hidden, 1)

    def nonlin(self, x):
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x, edge_index, node_type, edge_type, edge_attr):
        x, e = self.conv1(x, edge_index, node_type, edge_type, edge_attr)
        e = e.squeeze()
        x = self.nonlin(x)
        e = self.nonlin(e)

        x, e = self.conv2(x, edge_index, node_type, edge_type, e)
        e = e.squeeze()
        x = self.nonlin(x)
        e = self.nonlin(e)

        x, e = self.conv3(x, edge_index, node_type, edge_type, e)
        e = e.squeeze()
        x = self.nonlin(x)
        e = self.nonlin(e)

        x = self.node_classifier(x, node_type)
        e = self.edge_classifier(e, edge_type)
        return x, e


class HATGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden, device):
        super().__init__()
        self.device = device
        self.conv1 = EHATConv(node_feat, hidden, 3, 4, 4, edge_dim=1, edge_attr_emb_dim=hidden)
        # self.conv2 = EHATConv(hidden,    hidden, 3, 4, 4, edge_dim=hidden, edge_attr_emb_dim=hidden)
        # self.conv3 = EHATConv(hidden,    hidden, 3, 4, 4, edge_dim=hidden, edge_attr_emb_dim=hidden)
        # self.conv4 = EHATConv(hidden, hidden, 3, 4, 10, edge_dim=hidden, edge_attr_emb_dim=hidden)
        # self.conv5 = EHATConv(hidden, hidden, 3, 4, 10, edge_dim=hidden, edge_attr_emb_dim=hidden)
        # self.conv2 = EHATConv(hidden, hidden, 3, 4, 10, hidden, hidden)
        # self.conv3 = EHATConv(hidden, hidden, 3, 4, 10, hidden, hidden)

        self.node_classifier = HeteroLinear(hidden, 1, 3)
        self.edge_classifier = HeteroLinear(hidden, 1, 4)
        # self.classifier1 = nn.Linear(hidden, 1)

    def nonlin(self, x):
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x, edge_index, node_type, edge_type, edge_attr):
        x, e = self.conv1(x, edge_index, node_type, edge_type, torch.zeros(edge_type.shape[0], device=self.device).unsqueeze(1))
        e = e.squeeze()
        x = self.nonlin(x)
        e = self.nonlin(e)

        # x, e = self.conv2(x, edge_index, node_type, edge_type, e)
        # e = e.squeeze()
        # x = self.nonlin(x)
        # e = self.nonlin(e)
        # # #
        # x, e = self.conv3(x, edge_index, node_type, edge_type, e)
        # e = e.squeeze()
        # x = self.nonlin(x)
        # e = self.nonlin(e)

        # x, e = self.conv4(x, edge_index, node_type, edge_type, e)
        # e = e.squeeze()
        # x = self.nonlin(x)
        # e = self.nonlin(e)
        #
        # x, e = self.conv5(x, edge_index, node_type, edge_type, e)
        # e = e.squeeze()
        # x = self.nonlin(x)
        # e = self.nonlin(e)

        x = self.node_classifier(x, node_type)
        e = self.edge_classifier(e, edge_type)
        return x, e

class RESHATGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden):
        super().__init__()
        num_layer = 3
        self.conv1 = EHATConv(node_feat, hidden, 3, 4, 4, edge_dim=1, edge_attr_emb_dim=hidden)
        self.conv2 = EHATConv(hidden + node_feat, hidden, 3, 4, 4, edge_dim=hidden, edge_attr_emb_dim=hidden)
        self.conv3 = EHATConv(2 * hidden + node_feat, hidden, 3, 4, 4, edge_dim=2*hidden, edge_attr_emb_dim=hidden)
        # self.conv4 = EHATConv(3 * hidden + node_feat, hidden, 3, 4, 4, edge_dim=3*hidden, edge_attr_emb_dim=hidden)
        # self.conv5 = EHATConv(4 * hidden + node_feat, hidden, 3, 4, 4, edge_dim=4*hidden, edge_attr_emb_dim=hidden)
        # self.conv2 = EHATConv(hidden, hidden, 3, 4, 4, hidden, hidden)
        # self.conv3 = EHATConv(hidden, hidden, 3, 4, 4, hidden, hidden)

        self.node_classifier = HeteroLinear(num_layer*hidden + node_feat, 1, 3)
        self.edge_classifier = HeteroLinear(num_layer*hidden, 1, 4)
        # self.classifier1 = nn.Linear(hidden, 1)

    def nonlin(self, x):
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x, edge_index, node_type, edge_type, edge_attr):
        x1, e1 = self.conv1(x, edge_index, node_type, edge_type, torch.zeros(edge_type.shape[0]).unsqueeze(1))
        e1 = e1.squeeze()
        x1 = self.nonlin(x1)
        e1 = self.nonlin(e1)

        x2, e2 = self.conv2(torch.cat([x, x1], axis=1), edge_index, node_type, edge_type, e1)
        e2 = e2.squeeze()
        x2 = self.nonlin(x2)
        e2 = self.nonlin(e2)

        x3, e3 = self.conv3(torch.cat([x, x1, x2], axis=1), edge_index, node_type, edge_type, torch.cat([e1, e2], axis=1))
        e3 = e3.squeeze()
        x3 = self.nonlin(x3)
        e3 = self.nonlin(e3)

        # x4, e4 = self.conv4(torch.cat([x, x1, x2, x3], axis=1), edge_index, node_type, edge_type, torch.cat([e1, e2, e3], axis=1))
        # e4 = e4.squeeze()
        # x4 = self.nonlin(x4)
        # e4 = self.nonlin(e4)
        #
        # x5, e5 = self.conv4(torch.cat([x, x1, x2, x3, x4], axis=1), edge_index, node_type, edge_type, torch.cat([e1, e2, e3, e4], axis=1))
        # e5 = e5.squeeze()
        # x5 = self.nonlin(x5)
        # e5 = self.nonlin(e5)


        xl = self.node_classifier(torch.cat([x, x1, x2, x3], axis=1), node_type)
        el = self.edge_classifier(torch.cat([e1, e2, e3], axis=1), edge_type)
        # xl = self.node_classifier(torch.cat([x, x1, x2, x3, x4, x5], axis=1), node_type)
        # el = self.edge_classifier(torch.cat([e1, e2, e3, e4, e5], axis=1), edge_type)
        return xl, el

class HEATGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden):
        super().__init__()
        self.conv1 = EHEATConv(node_feat, hidden, 3, 4, 4, edge_feat, hidden)
        self.conv2 = EHEATConv(hidden, hidden, 3, 4, 4, edge_feat, hidden)
        self.conv3 = EHEATConv(hidden, hidden, 3, 4, 4, edge_feat, hidden)

        self.node_classifier = HeteroLinear(hidden, 1, 3)
        self.edge_classifier = HeteroLinear(hidden, 1, 4)
        # self.classifier1 = nn.Linear(hidden, 1)

    def nonlin(self, x):
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x, edge_index, node_type, edge_type, edge_attr):
        x, e = self.conv1(x, edge_index, node_type, edge_type, edge_attr)
        # e = e.squeeze()
        x = self.nonlin(x)
        # e = self.nonlin(e)

        x, e = self.conv2(x, edge_index, node_type, edge_type, edge_attr)
        # e = e.squeeze()
        x = self.nonlin(x)
        # e = self.nonlin(e)

        x, e = self.conv3(x, edge_index, node_type, edge_type, edge_attr)
        e = e.squeeze()
        x = self.nonlin(x)
        e = self.nonlin(e)

        x = self.node_classifier(x, node_type)
        e = self.edge_classifier(e, edge_type)
        return x, e
