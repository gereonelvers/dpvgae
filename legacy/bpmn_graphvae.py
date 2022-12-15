import os.path as osp
from random import random

import pm4py
import torch
import torch_geometric

import torch_geometric.transforms as T
from torch import FloatTensor
from torch.nn import ReLU, BCELoss, MSELoss, CrossEntropyLoss, NLLLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv, Linear
import time

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from bpmn_dataset import ProcessModelBPMNDataset
import pm4py.visualization.bpmn.visualizer as bpmn_visualizer
import wandb


# Probably not needed
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


def adj_to_edge_index(adj_matrix):
    edge_index = adj_matrix.nonzero().t().contiguous()
    #print("Edge Index:")
    #print(edge_index.shape)
    return edge_index


def kl_loss(self, mu=None, logstd=None):
    mu = self.__mu__ if mu is None else mu
    logstd = self.__logstd__ if logstd is None else logstd.clamp(
        max=10)
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


def loss_fn(adjacency_matrix, x_hat, graph):
    #random_tensor = torch.randn_like(adjacency_matrix)
    #rounded = torch.where((adjacency_matrix > random_tensor), 1, 0)
    # print(rounded)
    adapted_loss = torch.where(graph.adj_matrix_label.to(device) == 0, 1, 1)
    # print(adapted_loss)

    loss_function = BCELoss(weight=adapted_loss)  #, reduction="sum")
    node_loss_function = MSELoss()  #CrossEntropyLoss()
    # edge_loss = loss_function(adjacency_matrix, graph.adj_matrix_label.to(device))  # TODO
    edge_loss = loss_function(adjacency_matrix, graph.adj_matrix.to(device))
    # TODO:
    # 1. Make false 1s much more punishing in loss until no more 1s in prediction
    # 2. Slowly scale back until smth works with current dataset
    # 3. Later on: Generalize (learnable parameter?)
    kl_loss = model.kl_loss()
    # print(edge_loss)
    # print(kl_loss)

    autoencoder_loss = edge_loss + (1 / graph.num_nodes) * kl_loss
    # print(x_hat.shape)
    node_loss = node_loss_function(x_hat, graph.y.to(device))
    # additional_losses = torch.count_nonzero(rounded)
    # additional_losses = torch.count_nonzero(rounded)/len(rounded[0])
    # additional_losses = torch.abs((rounded - 0.5)%1)
    # print(additional_losses)

    # print(additional_losses)
    # print(node_loss)
    # print(autoencoder_loss)

    #print(autoencoder_loss)
    #print(node_loss)

    loss = 10*node_loss + autoencoder_loss #+ additional_losses #+ node_loss
    # print(loss)

    return node_loss


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Might need to define node_dim here (is 1 in petri net version)
        # Generate a random graph
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        # For node embeddings
        # https://github.com/pyg-team/pytorch_geometric/issues/2103#issuecomment-777295190
        self.conv_node = GCNConv(-1, 3)
        self.conv_node2 = GCNConv(3, 6)

    def forward(self, x, edge_index):
        x_new = torch.relu(self.conv1(x, edge_index))
        # fully_connected = torch_geometric.utils.dense_to_sparse(
        #     torch.ones(process_model_dataset.max_dim, process_model_dataset.max_dim))[0]
        return self.conv_mu(x_new, edge_index), self.conv_logstd(x_new, edge_index), self.conv_node2(torch.relu(
            self.conv_node(x, edge_index)), edge_index)


# class SimpleNodeGCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_node = GCNConv(-1, 3)
#         self.conv_node2 = GCNConv(3, 6)
#
#     def forward(self, x, edge_index):
#         return torch.sigmoid(self.conv_node2(torch.relu(self.conv_node(x, edge_index)), edge_index))
#
#
# class SimpleEdgeGCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conf_edges = GCNConv(-1, out_channels*2)
#         self.conf_edges2 = GCNConv(out_channels*2, out_channels)
#
#     def forward(self, x, edge_index):
#         return torch.sigmoid(self.conf_edges2(torch.relu(self.conf_edges(x, edge_index)), edge_index))


# This is just InnerProductDecoder from torch_geometric.nn.models.autoencoder
class Decoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


# Also stolen from torch_geometric.nn.models.autoencoder
class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        #return mu + torch.randn_like(logstd) * torch.exp(logstd)
        # if self.training:
        #     return mu + torch.randn_like(logstd) * torch.exp(logstd)
        # else:
        #     return mu
        return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__, x_out = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        # print("Params:")
        # print("mu: "+str(self.__mu__))
        # print("logstd: "+str(self.__logstd__))
        # print("z: "+str(z))
        return z, x_out

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=10)
        #print("mu")
        #print(mu)
        #print("logstd")
        #print(logstd)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


def train():
    losses = []
    # print("Training on " + str(len(train_data)) + " graphs")
    # for graph in train_data:
    #     model.train()
    #     optimizer.zero_grad()
    #     z, x_hat = model.encode(graph.x, graph.edge_index)
    #     output = model.decoder.forward_all(z)
    #
    #     # model_node.train()
    #     # model_edge.train()
    #     #
    #     # output = model_edge.forward(graph.x, graph.edge_index)
    #     # x_hat = model_node.forward(graph.x, graph.edge_index)
    #
    #     loss = loss_fn(output, x_hat, graph)
    #     losses.append(loss)
    graph = train_data[0]
    model.train()
    optimizer.zero_grad()
    z, x_hat = model.encode(graph.x, graph.edge_index)
    output = model.decoder.forward_all(z)
    loss = loss_fn(output, x_hat, graph)

    losses.append(loss)

    first = True
    if first and ([100, 200, 300, 398, 3998].__contains__(epoch)):
        first = False
        output = torch.where((output > 0.5), 1, 0)
        #process_model_dataset.plot_adjacency_matrix(graph.adj_matrix)
        #process_model_dataset.plot_adjacency_matrix(output)

        max_a, ids = torch.max(x_hat, 1, keepdim=True)
        x_out = torch.zeros_like(x_hat)
        x_out.scatter_(1, ids, max_a)
        x_out[x_out != 0] = 1

        #process_model_dataset.plot_adjacency_matrix(output)
        #bpmn = to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=graph.names))
        #pm4py.view_bpmn(bpmn)

        next_graph = train_data[2]
        #bpmn = to_bpmn(Data(x=next_graph.x, edge_index=adj_to_edge_index(output), names=next_graph.names))
        #pm4py.view_bpmn(bpmn)

    #total_loss = sum(losses)
    total_loss = sum(losses)
    total_loss.backward()
    optimizer.step()
    # node_optimizer.step()
    # edge_optimizer.step()

    return float(total_loss) #/ len(train_data)


@torch.no_grad()
def test(test_data):
    model.eval()
    losses = []
    # first = True
    # for graph in test_data:
    #     z, x_hat = model.encode(graph.x, graph.edge_index)
    #     output = model.decoder.forward_all(z)
    #
    #     # output = model_edge.forward(graph.x, graph.edge_index)
    #     # x_hat = model_node.forward(graph.x, graph.edge_index)
    #
    #     loss = loss_fn(output, x_hat, graph)
    #     losses.append(loss)
    #
    #     # # TODO: ugly and hacky
    #     # #first = False
    #     # if first and ([1,20,100,200,300,398,3998].__contains__(epoch)):
    #     #     first = False
    #     #     # TODO: Finish visualization. This should log to tensorboard/an alternative - not pop up on desktop. Nodes with diagonal below threshold should be deleted.
    #     #     # TODO: The fact that none of these things look like process models shows that whatever we are currently learning is BS. Figure out why.
    #     #     # - Delete nodes that should not exist
    #     #     # - Remove all edges that point to a deleted node
    #     #     # Maybe the one-hot stuff from here should be considered for loss function?
    #     #
    #     #     max_a, ids = torch.max(x_hat, 1, keepdim=True)
    #     #     x_out = torch.zeros_like(x_hat)
    #     #     x_out.scatter_(1, ids, max_a)
    #     #     x_out[x_out != 0] = 1
    #     #
    #     #     #print(torch.round(output, decimals=4))
    #     #     #print(torch.mean(output))
    #     #     random_tensor = torch.randn_like(output)
    #     #     output = torch.where((output > 0.50), 1, 0)
    #     #     print(torch.count_nonzero(output))
    #     #     #output.fill_diagonal_(0)
    #     #
    #     #     #process_model_dataset.plot_adjacency_matrix(graph.adj_matrix_label)
    #     #     process_model_dataset.plot_adjacency_matrix(output)
    #     #     #print(output)
    #     #
    #     #     #bpmn = to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=graph.names_labels))
    #     #     #pm4py.view_bpmn(bpmn)
    #     #     #print(x_out)
    #     #     #print(graph.y)
    #     #
    #     #     #bpmn = to_bpmn(Data(x=graph.x, edge_index=graph.edge_index, names=graph.names))
    #     #     #pm4py.view_bpmn(bpmn)
    #     #     #bpmn_visualizer.save(bpmn_visualizer.apply(bpmn),
    #     #     #                       osp.join(path, "visualizations", "bpmn_{}.png".format(epoch)))
    #     #     #writer.add_figure("{}/BPMN".format(epoch), osp.join(path, "visualizations", "bpmn_{}.png".format(epoch)))
    #     #     #pm4py.view_bpmn(bpmn)
    graph = test_data[0]
    z, x_hat = model.encode(graph.x, graph.edge_index)
    output = model.decoder.forward_all(z)
    loss = loss_fn(output, x_hat, graph)
    losses.append(loss)

    first = True
    if first and ([10000].__contains__(epoch)):
        first = False


        output = torch.where((output > 0.70), 1, 0)
        max_a, ids = torch.max(x_hat, 1, keepdim=True)
        x_out = torch.zeros_like(x_hat)
        x_out.scatter_(1, ids, max_a)
        x_out[x_out != 0] = 1
        print(x_out)
        print(graph.y)
        #process_model_dataset.plot_adjacency_matrix(output)
        #bpmn = to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=graph.names))
        #pm4py.view_bpmn(bpmn)


    return sum(losses) #/ len(test_data)


# Main idea: https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129
if __name__ == '__main__':
    epochs = 25000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data-dump', 'bpmn_graphvae')
    writer = SummaryWriter(log_dir=path + '/logs/' + time.strftime("%Y%m%d-%H%M%S"))
    # wandb.init(project="bpmn-graphvae")

    transform = T.Compose([
        T.ToDevice(device),
    ])
    process_model_dataset = ProcessModelBPMNDataset(path, transform=transform)

    # process_model_dataset.plot_adjacency_matrix(process_model_dataset.get(0).adj_matrix)
    # process_model_dataset.plot_adjacency_matrix(process_model_dataset.get(10).adj_matrix)
    # process_model_dataset.plot_adjacency_matrix(process_model_dataset.get(20).adj_matrix)
    # process_model_dataset.plot_adjacency_matrix(process_model_dataset.get(30).adj_matrix)
    # process_model_dataset.plot_adjacency_matrix(process_model_dataset.get(40).adj_matrix)


    # print("Loss Test:")
    # label = torch.FloatTensor([[0,0],[0,0]])
    # adj_matrix = torch.FloatTensor([[1,1],[1,1]])
    #
    # adapted_loss = torch.where(label.to(device) <= 0, 1000, 1)
    # print(adapted_loss)
    #
    # loss_function = BCELoss(reduction="sum", weight=adapted_loss)#, reduction="sum")
    # loss = loss_function(adj_matrix, label)
    # print(str(loss))
    # exit(1)

    train_data = process_model_dataset.get_train_data()
    test_data = process_model_dataset.get_test_data()
    val_data = process_model_dataset.get_val_data()

    in_channels, out_channels = -1, process_model_dataset.max_dim

    torch.set_printoptions(threshold=10_000, sci_mode=False)
    model = VGAE(encoder=Encoder(in_channels, out_channels), decoder=Decoder()).to(device)
    #model = VGAE(encoder=Encoder(in_channels, out_channels)).to(device)
    #model_node = SimpleNodeGCN().to(device)
    #model_edge = SimpleEdgeGCN().to(device)

    print("Dataset Max Dim: " + str(process_model_dataset.max_dim))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #, weight_decay=0.1)

    #node_optimizer = torch.optim.Adam(model_node.parameters(), lr=0.001)
    #edge_optimizer = torch.optim.Adam(model_edge.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        loss = train()
        test_loss = test(test_data)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        # wandb.log({"loss/train": loss})
        # wandb.log({"loss/test": test_loss})
        # Log graph to tensorboard
        # writer.add_figure("Graph-{}".format(epoch), process_model_dataset.draw_graph(process_model_dataset.get(0)))  # TODO: Sample and log correct bpmn models here
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_loss:.4f}')

