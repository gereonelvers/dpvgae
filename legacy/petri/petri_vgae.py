import argparse
import os.path as osp

import networkx as nx
import torch

import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, GCNConv
import time

from torch_geometric.utils import negative_sampling

from petri_dataset import ProcessModelPetriDataset
from petri_in_memory_dataset import ProcessModelMemoryDataset


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Defining node_dim here is hacky and might be completely wrong!
        self.conv1 = GCNConv(in_channels, 2 * out_channels, node_dim=0)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, node_dim=0)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, node_dim=0)
        # self.conv1 = GCNConv(in_channels, 2 * out_channels)
        # self.conv_mu = GCNConv(2 * out_channels, out_channels)
        # self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def convert_edge_index(edge_index):
    print("Input edge_index:")
    print(edge_index)
    mapping = {}
    mapped_edge_index = []
    for (src, dst) in edge_index.t().tolist():
        if src not in mapping:
            mapping[src] = len(mapping)
        if dst not in mapping:
            mapping[dst] = len(mapping)
        mapped_edge_index.append([mapping[src], mapping[dst]])
    edge_index = torch.tensor(mapped_edge_index).t().contiguous()
    print("Output edge_index:")
    print(edge_index)
    return edge_index


def train():
    # model.train()
    # optimizer.zero_grad()
    # z = model.encode(train_data.x, train_data.edge_index)
    # # TODO: Once dataset works with this, adapt loss function:
    # # Step 1:
    # # - Validate formal correctness of petri net
    # # Step 2:
    # # - Process Redesign KPI stuff +
    # # - Process Model Complexity stuff
    # # loss = model.recon_loss(z, train_data.pos_edge_label_index)
    # loss = model.recon_loss(z, train_data.edge_index_labels)
    # loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    # # print("Loss:")
    # # print(loss)
    # # type(loss)
    # # print(train_data.pos_edge_label_index)
    # loss.backward()
    # optimizer.step()
    # return float(loss)
    for graph in train_data:
        model.train()
        optimizer.zero_grad()
        z = model.encode(graph.x, graph.edge_index)
        # TODO: Once dataset works with this, adapt loss function:
        # Step 1:
        # - Validate formal correctness of petri net
        # Step 2:
        # - Process Redesign KPI stuff +
        # - Process Model Complexity stuff
        loss = model.recon_loss(z, train_data.pos_edge_label_index)

        # Loss function based on VGAE
        # output = model.decode(z, torch.zeros(graph.num_nodes, 1, dtype=torch.float))
        # print("Output:")
        # print(output)
        #
        # loss = model.recon_loss(z, graph.edge_index_labels) + (1 / graph.num_nodes) * model.kl_loss()
        # print("Loss:")
        # print(loss)
        # print(type(loss))

        # Basic loss function:
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(z[graph.edge_index_labels[0]], graph.edge_index_labels[1].float())


        loss = loss + (1 / graph.num_nodes) * model.kl_loss()
        # print("Loss:")
        # print(loss)
        # type(loss)
        # print(train_data.pos_edge_label_index)
        loss.backward()
        optimizer.step()
        return float(loss)


@torch.no_grad()
def test(data):
    # model.eval()
    # z = model.encode(process-datasets.x, process-datasets.edge_index)
    # return model.test(z, process-datasets.pos_edge_label_index, process-datasets.neg_edge_label_index)
    auc, ap = 0, 0
    for graph in data:
        model.eval()
        # print("Graph:")
        # print(graph)
        z = model.encode(graph.x, graph.edge_index)
        mAuc, mAp = model.test(z, graph.edge_index, None)
        auc, ap = mAuc + auc, mAp + ap
    # return model.test(z, process-datasets.pos_edge_label_index, process-datasets.neg_edge_label_index)
    # print(process-datasets.edge_index)
    return auc, ap


MAX_LOGSTD = 10


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='Cora',
    #                     choices=['Cora', 'CiteSeer', 'PubMed'])
    # parser.add_argument('--dataset', type=str, default='PubMed',
    #                     choices=['Cora', 'CiteSeer', 'PubMed'])
    # parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../process-vgae-trad')
    writer = SummaryWriter(log_dir=path+'/logs/' + time.strftime("%Y%m%d-%H%M%S"))

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    # dataset = Planetoid(path, args.dataset, transform=transform)
    # train_data, val_data, test_data = dataset[0]
    in_memory_dataset = ProcessModelMemoryDataset(path, transform=transform)


    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])
    process_model_dataset = ProcessModelPetriDataset(path, transform=transform)

    # print(process_model_dataset[0])
    # train_data = process_model_dataset[0]
    # val_data = process_model_dataset[1]
    # test_data = process_model_dataset[2]

    # train_data = in_memory_dataset.collate(process_model_dataset.get_train_data())[0]
    # test_data = in_memory_dataset.collate(process_model_dataset.get_test_data())[0]
    # val_data = in_memory_dataset.collate(process_model_dataset.get_val_data())[0]
    train_data = process_model_dataset.get_train_data()
    test_data = process_model_dataset.get_test_data()
    val_data = process_model_dataset.get_val_data()
    # # train_data, val_data, test_data = in_memory_dataset[0]
    # for graph in test_data:
    #     assert convert_edge_index(graph.edge_index).max() > graph.num_nodes
            # print('edge index max is not greater than num nodes')
            # print(graph.edge_index.max())

    # train_data = [process_model_dataset.get(0)]
    # test_data = [process_model_dataset.get(0)]
    # val_data = [process_model_dataset.get(0)]


    in_channels, out_channels = -1, 25

    torch.set_printoptions(threshold=10_000)
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        loss = train()
        auc, ap = test(test_data)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('AUC/test', auc, epoch)
        writer.add_scalar('AP/test', ap, epoch)
        # Log graph to tensorboard
        # writer.add_figure("Graph-{}".format(epoch), process_model_dataset.draw_graph(process_model_dataset.get(0)))  # TODO: Sample and log correct graph here
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')

    # TODO: Figure out how to get model to output something useful :)
    model.eval()
    z = model.encode(process_model_dataset.get(99).x, process_model_dataset.get(99).edge_index)
    output = model.decode(z, process_model_dataset.get(99).edge_index)
    print(output)
    print(process_model_dataset.get(99).edge_index_labels)
