# import matplotlib.pyplot as plt
# n_agent = 5
# random_win_rate = [1 / n_agent] * 20
#
# # my_win_rate = [10, 10, 10, 10, 12, 15, 19, 17, 20, 20, 20, 20, 18, 19, 19, 16, 16, 7, 3, 6]
# # my_win_rate = [10, 10, 10, 10, 10, 10, 18, 22, 19, 17, 21, 19, 18, 14, 18, 17, 15, 15, 11, 5]
# # my_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 19, 21, 24, 22, 28, 25, 28, 23, 23, 18, 14, 14]
# my_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 23, 25, 25, 22, 27, 27, 18, 20, 18, 12]
#
# # new_win_rate = [10, 10, 10, 10, 16, 18, 15, 19, 19, 20, 20, 20, 18, 17, 19, 20, 13, 14, 19, 9]
# # new_win_rate = [10, 10, 10, 10, 10, 10, 12, 21, 22, 21, 23, 21, 22, 29, 25, 24, 25, 16, 6, 9]
# # new_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 26, 21, 22, 23, 25, 29, 33, 21, 26, 12, 13, 15]
# new_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 27, 26, 33, 36, 25, 21, 33, 16, 17, 18]
#
# # mc_win_rate = [10, 10, 10, 10, 17, 18, 20, 20, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# # mc_win_rate = [10, 10, 10, 10, 10, 10, 26, 22, 24, 21, 27, 25, 30, 28, 27, 29, 30, 29, 30, 28]
# # mc_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 24, 27, 28, 30, 31, 34, 36, 34, 34, 37, 35, 35]
# mc_win_rate = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 33, 40, 25, 34, 37, 42, 35, 39, 40, 39]
#
#
# norm_win = [win_rate / 10 / n_agent for win_rate in my_win_rate]
# norm_mcs = [win_rate / 10 / n_agent for win_rate in mc_win_rate]
# norm_new = [win_rate / 10 / n_agent for win_rate in new_win_rate]
#
# plt.plot(random_win_rate, label='Random')
# plt.plot(norm_win, label='RL - 100')
# plt.plot(norm_new, label='RL - 300')
# plt.plot(norm_mcs, label='MCTS')
# plt.title('Win Rate in {} Player Game'.format(n_agent))
# plt.ylabel('Win Rate')
# plt.xlabel('Board Size')
# plt.xticks([i for i in range(21)])
# plt.legend()
# plt.show()

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero

import torch

data = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())[0]
num_classes = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected()).num_classes

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
print(model)
