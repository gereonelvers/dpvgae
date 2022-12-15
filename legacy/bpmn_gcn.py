import os.path as osp

import pm4py
import torch

import torch_geometric.transforms as T
from pm4py.objects.bpmn.obj import BPMN
from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import time

from bpmn_dataset import ProcessModelBPMNDataset
from legacy.bpmn_graphvae import adj_to_edge_index


def loss_fn(adjacency_matrix, x_hat, graph):
    # Loss functions
    loss_function = BCELoss()
    node_loss_function = CrossEntropyLoss()
    # Edge loss
    # edge_loss = loss_function(adjacency_matrix, graph.adj_matrix_label.to(device))
    edge_loss = loss_function(adjacency_matrix, graph.adj_matrix.to(device))
    kl_loss_edge = model.kl_loss(adjacency_matrix, graph.adj_matrix_label.to(device))
    # edge_loss += + (1 / process_model_dataset.max_dim) * kl_loss_edge  # TODO
    # Node loss
    node_loss = node_loss_function(x_hat, torch.argmax(graph.y, dim=1).to(device))
    kl_loss_node = model.kl_loss(x_hat, graph.y.to(device))
    node_loss += (1 / process_model_dataset.max_dim) * kl_loss_node
    return node_loss, edge_loss


class BPMNSimpleModel(torch.nn.Module):
    def __init__(self, node_channels, edge_channels):
        super().__init__()
        # Edge layers
        self.conv_1_edge = GATConv(node_channels, round(node_channels+(edge_channels-node_channels)*0.3))  # +3
        self.conv_2_edge = GATConv(round(node_channels+(edge_channels-node_channels)*0.3), round(node_channels+(edge_channels-node_channels)*0.6))  # +4
        self.conv_3_edge = GATConv(round(node_channels+(edge_channels-node_channels)*0.6), round(node_channels+(edge_channels-node_channels)*0.8))  # +5
        # self.conv_4_edge = GATConv(round(node_channels+(edge_channels-node_channels)*0.6), round(node_channels+(edge_channels-node_channels)*0.8))  # +6
        #self.conv_5_edge = GATConv(78, edge_channels)  # +6
        #self.conv_6_edge = GATConv(40, 45)  # +7
        #self.conv_7_edge = GATConv(45, 52)  # +7
        #self.conv_8_edge = GATConv(52, 59)  # +8 (should be)
        #self.conv_9_edge = GATConv(59, 64)  # +8 (should be)
        #self.conv_10_edge = GATConv(64, 70)  # +8 (should be)
        #self.conv_11_edge = GATConv(70, 78)# +8 (should be)
        #self.conv_12_edge = GATConv(78, 84)
        self.conv_mu_edge = GATConv(round(node_channels+(edge_channels-node_channels)*0.8), edge_channels)
        self.conv_logstd_edge = GATConv(round(node_channels+(edge_channels-node_channels)*0.8), edge_channels)
        # Node layers
        self.conv1_node = GCNConv(node_channels, 2 * node_channels)
        self.conv2_node = GCNConv(2 * node_channels, node_channels)
        self.conv3_node = GCNConv(node_channels, 2 * node_channels)
        self.conv_mu_node = GCNConv(2 * node_channels, node_channels)
        self.conv_logstd_node = GCNConv(2 * node_channels, node_channels)

    def encode_edge(self, x, edge_index):
        # print("Encode edge with x shape: ", x.shape)
        # print("While edge index shape is: ", edge_index.shape)
        edges = self.conv_1_edge(x, edge_index).relu()
        edges = self.conv_2_edge(edges, edge_index).relu()
        edges = self.conv_3_edge(edges, edge_index).relu()
        # edges = self.conv_4_edge(edges, edge_index).relu()
        #edges = self.conv_5_edge(edges, edge_index).relu()
        #edges = self.conv_6_edge(edges, edge_index).relu()
        #edges = self.conv_7_edge(edges, edge_index).relu()
        #edges = self.conv_8_edge(edges, edge_index).relu()
        #edges = self.conv_9_edge(edges, edge_index).relu()
        #edges = self.conv_10_edge(edges, edge_index).relu()
        #edges = self.conv_11_edge(edges, edge_index).relu()
        #edges = self.conv_12_edge(edges, edge_index).relu()

        conv_mu_edge = self.conv_mu_edge(edges, edge_index)
        conv_logstd_edge = self.conv_logstd_edge(edges, edge_index)
        # print("Conv mu edge shape: ", conv_mu_edge.shape)
        # print("Conv logstd edge shape: ", conv_logstd_edge.shape)
        return conv_mu_edge, conv_logstd_edge

    def decode_edge(self, x, z, sigmoid=True):
        adj = z  # TODO: consider better decoder architecture
        # print("Decode edge with x shape: ", x.shape)
        # print("While z shape is: ", z.shape)
        return torch.sigmoid(adj) if sigmoid else adj

    def encode_node(self, x, edge_index):
        nodes = self.conv1_node(x, edge_index).relu()
        nodes = self.conv2_node(nodes, edge_index).relu()
        nodes = self.conv3_node(nodes, edge_index).relu()
        return self.conv_mu_node(nodes, edge_index), self.conv_logstd_node(nodes, edge_index)

    def decode_node(self, x, z, sigmoid=True):
        adj = z  # TODO: consider better decoder architecture
        return torch.sigmoid(adj) if sigmoid else adj

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu  # + torch.div(torch.randn_like(logstd) * torch.exp(logstd), 10)  # TODO
        else:
            return mu

    def kl_loss(self, mu, logstd):
        logstd = logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def forward(self, x, edge_index):
        mu_edge, logstd_edge = self.encode_edge(x, edge_index)
        z = self.reparametrize(mu_edge, logstd_edge)
        # print("Since decoder only sigmoids, z shape (which is used as edge) is: ", z.shape)
        edges = self.decode_edge(x, z)

        mu_node, logstd_node = self.encode_node(x, edge_index)
        z = self.reparametrize(mu_node, logstd_node)
        nodes = self.decode_node(x, z)
        return edges, nodes


def train():
    model.train()
    node_optimizer.zero_grad()
    edge_optimizer.zero_grad()
    node_losses = []
    edge_losses = []
    first = True
    for graph in train_data:
        graph = train_data[0]
        adj_hat, x_hat = model(graph.x.to(device), graph.edge_index.to(device))

        if first and ([300, 1198, 1500, 1998, 4000, 6000, 8000, 9998].__contains__(epoch)):
            first = False
            visualization(graph, adj_hat, x_hat)

        node_loss, edge_loss = loss_fn(adj_hat, x_hat, graph)
        node_losses.append(node_loss)
        edge_losses.append(edge_loss)
    total_node_loss = sum(node_losses)
    total_edge_loss = sum(edge_losses)
    total_node_loss.backward()
    total_edge_loss.backward()
    node_optimizer.step()
    edge_optimizer.step()
    return float(total_node_loss) / len(train_data), float(total_edge_loss) / len(train_data)


@torch.no_grad()
def test():
    model.eval()
    node_losses = []
    edge_losses = []
    for graph in test_data:
        graph = test_data[0]
        adj_hat, x_hat = model(graph.x.to(device), graph.edge_index.to(device))
        node_loss, edge_loss = loss_fn(adj_hat, x_hat, graph)
        node_losses.append(node_loss)
        edge_losses.append(edge_loss)
    return sum(node_losses) / len(test_data), sum(edge_losses) / len(test_data)


def visualization(graph, adj_hat, x_hat):
    # -- nodes --
    # print(x_hat)
    # print(x_hat.shape)
    max_a, ids = torch.max(x_hat, 1, keepdim=True)
    x_out = torch.zeros_like(x_hat)
    x_out.scatter_(1, ids, max_a)
    x_out[x_out != 0] = 1
    # process_model_dataset.plot_node_features(graph.y.t(), title="Node labels for epoch " + str(epoch))
    # process_model_dataset.plot_node_features(x_hat.detach().cpu().t().numpy(),
    #                                          title="Node predictions for epoch " + str(epoch))
    # process_model_dataset.plot_node_features(x_out.detach().cpu().t().numpy(),
    #                                         title="Max node prediction for epoch " + str(epoch))
    # process_model_dataset.plot_node_features(x_out.detach().cpu().t().numpy())

    # -- edges --
    # print(adj_hat)
    # print(adj_hat.shape)
    output = torch.where((adj_hat > 0.5), 1, 0)
    process_model_dataset.plot_adjacency_matrix(graph.adj_matrix_label,
                                                title="Edge labels for epoch " + str(epoch) + " in run " + str(
                                                    run_name))
    process_model_dataset.plot_adjacency_matrix(adj_hat.detach().cpu().numpy(),
                                                title="Edge predictions for epoch " + str(epoch) + " in run " + str(
                                                    run_name))
    # process_model_dataset.plot_adjacency_matrix(output.detach().cpu().numpy())

    # -- General --
    output.diagonal().fill_(0)
    bpmn = process_model_dataset.to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=graph.names_labels))
    pm4py.view_bpmn(bpmn)


if __name__ == '__main__':
    run_name = "cdlg-ss-no-logstd-no-kl-scaleup-4-layer-scaling-gatconv-long-training"

    epochs = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    torch.set_printoptions(threshold=10_000, sci_mode=False)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data-dump', 'bpmn_graphvae')
    writer = SummaryWriter(log_dir=path + '/logs/' + (time.strftime("%Y%m%d-%H%M%S") + "-" + run_name))
    # wandb.init(project="bpmn-graphvae")

    transform = T.ToDevice(device)
    process_model_dataset = ProcessModelBPMNDataset(path, transform=transform)
    train_data = process_model_dataset.get_train_data(shuffle=False)
    test_data = process_model_dataset.get_test_data(shuffle=False)
    val_data = process_model_dataset.get_val_data(shuffle=False)

    model = BPMNSimpleModel(node_channels=6, edge_channels=process_model_dataset.max_dim).to(device)

    print("Dataset Max Dim: " + str(process_model_dataset.max_dim))

    model = model.to(device)
    edge_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # for edges
    node_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Visulize first graph
    bpmn = process_model_dataset.to_bpmn(train_data[0])
    pm4py.view_bpmn(bpmn)

    for epoch in range(1, epochs + 1):
        node_loss, edge_loss = train()
        test_node_loss, test_edge_loss = test()
        writer.add_scalar('Loss/train/node', node_loss, epoch)
        writer.add_scalar('Loss/train/edge', edge_loss, epoch)
        writer.add_scalar('Loss/test/node', test_node_loss, epoch)
        writer.add_scalar('Loss/test/edge', test_edge_loss, epoch)
        print(
            'Epoch: {:03d}, Train Node Loss: {:.7f}, Train Edge Loss: {:.7f}, Test Node Loss: {:.7f}, Test Edge Loss: {:.7f}'.format(
                epoch, node_loss, edge_loss, test_node_loss, test_edge_loss))
