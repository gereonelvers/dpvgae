import torch

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from bpmn_dataset import ProcessModelBPMNDataset

#
#
# class VariationalGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels, normalize=False)
#         self.conv_mu = GCNConv(2 * out_channels, out_channels, normalize=False)
#         self.conv_logstd = GCNConv(2 * out_channels, out_channels, normalize=False)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
# #
# # class VariationalLinearEncoder(torch.nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.conv_mu = GCNConv(in_channels, out_channels)
# #         self.conv_logstd = GCNConv(in_channels, out_channels)
# #
# #     def forward(self, x, edge_index):
# #         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
#
# def train(process-datasets):
#     model.train()
#     optimizer.zero_grad()
#     loss = 0
#     for index, graph in enumerate(process-datasets):
#         z = model.encode(graph.x[:26], graph.edge_index[:26])
#         # loss = model.recon_loss(z, graph.pos_edge_label_index)
#         loss = model.recon_loss(z, graph.y.edge_index)
#         # loss = loss + (1 / graph.num_nodes) * model.kl_loss()
#     print(loss)
#     loss.backward()
#     optimizer.step()
#     return float(loss)
#
#
# @torch.no_grad()
# def test(process-datasets):
#     model.eval()
#     z = model.encode(process-datasets.x, process-datasets.edge_index)
#     return model.test(z, process-datasets.pos_edge_label_index, process-datasets.neg_edge_label_index)
#
#
# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--variational', action='store_true')
#     # parser.add_argument('--linear', action='store_true')
#     # parser.add_argument('--dataset', type=str, default='Cora',
#     #                     choices=['Cora', 'CiteSeer', 'PubMed'])
#     # parser.add_argument('--epochs', type=int, default=400)
#     # args = parser.parse_args()
#
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # transform = T.Compose([
#     #     T.NormalizeFeatures(),
#     #     T.ToDevice(device),
#     #     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#     #                       split_labels=True, add_negative_train_samples=False),
#     # ])
#
#     # dataset = Planetoid(path, args.dataset, transform=transform)
#     dataset = ProcessModelDataset("./process-datasets/process_dataset")
#     train_data = dataset[:60]
#     val_data = dataset[60:80]
#     test_data = dataset[80:]
#
#     in_channels, out_channels = 26, dataset.num_features
#     model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
#
#     # model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     for epoch in range(1, 200):
#         loss = train(train_data)
#         # loss = train()
#         # optimizer.zero_grad()
#         # out = model(process-datasets)
#         # # loss = F.nll_loss(out[process-datasets.train_mask], process-datasets.y[process-datasets.train_mask])
#         # # loss.backward()
#         # # optimizer.step()
#         # auc, ap = test(test_data)
#         # print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
#
#     # model.eval()
#     # pred = model(process-datasets).argmax(dim=1)
#     # correct = (pred[process-datasets.test_mask] == process-datasets.y[process-datasets.test_mask]).sum()
#     # acc = int(correct) / int(process-datasets.test_mask.sum())
#     # print(f'Accuracy: {acc:.4f}')

# import torch
# from torch_geometric.datasets import Planetoid
# from torchgen.context import F
#
# from process_model_dataset import ProcessModelDataset
# from process_model_drift_dataset import SimpleGCN
# from torch_geometric.nn.models import VGAE
#
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     dataset = Planetoid(root='/tmp/Cora', name='Cora')
#     process-datasets = dataset[0].to(device)
#     # loader = ProcessModelDataset(root="process-datasets/process_model_dataset")
#     # dataset = loader.get_dataset()
#
#     model = VGAE(encoder=SimpleGCN(27, 27), decoder=SimpleGCN(27, 27)).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#     model.train()
#     for epoch in range(200):
#         optimizer.zero_grad()
#         out = model(process-datasets)
#         loss = F.nll_loss(out[process-datasets.train_mask], process-datasets.y[process-datasets.train_mask])
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     pred = model(process-datasets).argmax(dim=1)
#     correct = (pred[process-datasets.test_mask] == process-datasets.y[process-datasets.test_mask]).sum()
#     acc = int(correct) / int(process-datasets.test_mask.sum())
#     print(f'Accuracy: {acc:.4f}')

EPS = 1e-15
MAX_LOGSTD = 10

# class ProcessVGAE(GAE):
    # r"""The Variational Graph Auto-Encoder model from the
    # `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    # paper.
    #
    # Args:
    #     encoder (Module): The encoder module to compute :math:`\mu` and
    #         :math:`\log\sigma^2`.
    #     decoder (Module, optional): The decoder module. If set to :obj:`None`,
    #         will default to the
    #         :class:`torch_geometric.nn.models.InnerProductDecoder`.
    #         (default: :obj:`None`)
    # """
    # def __init__(self, encoder, decoder=None):
    #     super().__init__(encoder, decoder)
    #
    # def reparametrize(self, mu, logstd):
    #     if self.training:
    #         return mu + torch.randn_like(logstd) * torch.exp(logstd)
    #     else:
    #         return mu
    #
    # def encode(self, *args, **kwargs):
    #     """"""
    #     self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
    #     self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
    #     z = self.reparametrize(self.__mu__, self.__logstd__)
    #     return z
    #
    # def kl_loss(self, mu=None, logstd=None):
    #     r"""Computes the KL loss, either for the passed arguments :obj:`mu`
    #     and :obj:`logstd`, or based on latent variables from last encoding.
    #
    #     Args:
    #         mu (Tensor, optional): The latent space for :math:`\mu`. If set to
    #             :obj:`None`, uses the last computation of :math:`mu`.
    #             (default: :obj:`None`)
    #         logstd (Tensor, optional): The latent space for
    #             :math:`\log\sigma`.  If set to :obj:`None`, uses the last
    #             computation of :math:`\log\sigma^2`.(default: :obj:`None`)
    #     """
    #     mu = self.__mu__ if mu is None else mu
    #     logstd = self.__logstd__ if logstd is None else logstd.clamp(
    #         max=MAX_LOGSTD)
    #     return -0.5 * torch.mean(
    #         torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1_nodes = GCNConv(-1, 2 * out_channels)
        self.conv2_nodes = GCNConv(2 * out_channels, out_channels)
        self.conv1_edges = GCNConv(-1, 2 * out_channels)
        self.conv2_edges = GCNConv(2 * out_channels, out_channels)


    def forward(self, x, edge_index):
        x = self.conv1_nodes(x, edge_index).relu()
        x = self.conv2_nodes(x, edge_index)
        edge_index = self.conv1_edges(x, edge_index).relu()
        edge_index = self.conv2_edges(x, edge_index)
        return x, edge_index


class ProcessVGAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(25, 50, node_dim=0, normalize=False)
        # self.conv2 = GCNConv(50, 25, node_dim=0, normalize=False)
        # self.conv2 = GCNConv(16, 8)
        # self.conv3 = GCNConv(8, 4)
        # self.conv4 = GCNConv(4, 8)
        # self.conv5 = GCNConv(8, 16)
        # self.conv6 = GCNConv(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv5(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv6(x, edge_index)

        return x  # F.log_softmax(x, dim=1)


if __name__ == '__main__':
    dataset = ProcessModelBPMNDataset("./process-datasets/process_dataset")
    train_loader = DataLoader(dataset[:60])
    val_loader = DataLoader(dataset[60:80])
    test_loader = DataLoader(dataset[80:])

    in_channels, out_channels = -1, dataset.max_dim

    model = VariationalGCNEncoder(in_channels, out_channels)
    # model = VariationalGCNEncoder(in_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 100):
        model.train()
        optimizer.zero_grad()
        for idx, graph in enumerate(train_loader):
            z = model.encode(graph.x[:25], graph.edge_index[0:2, :25])
            # z = model(graph.x[:25], graph.edge_index[0:2, :25])
            MSE = torch.nn.MSELoss()
            print(z)
            loss = MSE(z, graph.x_labels[:25])
            loss = model.recon_loss(z, graph.pos_edge_label_index)
            # loss = loss + (1 / len(graph.x)) * model.kl_loss()  # TODO

            loss.backward()
            optimizer.step()
            print(str(float(loss)))
