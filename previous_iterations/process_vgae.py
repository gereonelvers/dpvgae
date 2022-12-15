import os.path as osp

import pm4py
import torch

import torch_geometric.transforms as T
from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import time


from bpmn_dataset import ProcessModelBPMNDataset
from legacy.bpmn_graphvae import adj_to_edge_index


def loss_fn(adjacency_matrix, x_hat, graph):
    # Loss functions
    loss_function = BCELoss()
    node_loss_function = CrossEntropyLoss()
    # Edge los
    edge_loss = loss_function(adjacency_matrix, graph.adj_matrix_label.to(device))
    kl_loss_edge = model.kl_loss(adjacency_matrix, graph.adj_matrix_label.to(device))
    edge_loss += + (1 / process_model_dataset.max_dim) * kl_loss_edge

    # Node loss
    node_loss = node_loss_function(x_hat, torch.argmax(graph.y, dim=1).to(device))
    kl_loss_node = model.kl_loss(x_hat, graph.y.to(device))
    node_loss += (1 / process_model_dataset.max_dim) * kl_loss_node
    return node_loss, edge_loss


class ProcessVGAE(torch.nn.Module):
    def __init__(self, node_channels, edge_channels):
        super().__init__()
        # Edge layers
        self.conv_1_edge = GCNConv(node_channels, round(node_channels + (edge_channels - node_channels) * 0.3))
        self.conv_2_edge = GCNConv(round(node_channels + (edge_channels - node_channels) * 0.3),
                                   round(node_channels + (edge_channels - node_channels) * 0.5))

        self.conv_mu_edge = GCNConv(round(node_channels + (edge_channels - node_channels) * 0.5),
                                    round(node_channels + (edge_channels - node_channels) * 0.7))
        self.conv_logstd_edge = GCNConv(round(node_channels + (edge_channels - node_channels) * 0.5),
                                        round(node_channels + (edge_channels - node_channels) * 0.7))

        self.conv_3_edge = GCNConv(round(node_channels+(edge_channels-node_channels)*0.7), round(node_channels+(edge_channels-node_channels)*0.6))
        self.conv_4_edge = GCNConv(round(node_channels + (edge_channels - node_channels) * 0.7), edge_channels)

        # Node layers
        self.conv1_node = GCNConv(node_channels, node_channels)
        # self.conv2_node = GCNConv(node_channels, node_channels)
        # self.conv3_node = GCNConv(node_channels, node_channels)
        # self.conv4_node = GCNConv(node_channels, node_channels)
        self.conv_mu_node = GCNConv(node_channels, node_channels)
        self.conv_logstd_node = GCNConv(node_channels, node_channels)
        self.conv5_node = GCNConv(node_channels, node_channels)
        # self.conv6_node = GCNConv(node_channels, node_channels)
        # self.conv7_node = GCNConv(node_channels, node_channels)
        # self.conv8_node = GCNConv(node_channels, node_channels)

    def process_edges(self, x, edge_index):
        # print("Encode edge with x shape: ", x.shape)
        # print("While edge index shape is: ", edge_index.shape)
        edges = self.conv_1_edge(x, edge_index).relu()
        edges = self.conv_2_edge(edges, edge_index).relu()
        conv_mu_edge = self.conv_mu_edge(edges, edge_index)
        conv_logstd_edge = self.conv_logstd_edge(edges, edge_index)
        edges = self.reparametrize(conv_mu_edge, conv_logstd_edge)
        # edges = self.conv_3_edge(edges, edge_index).relu()
        edges = self.conv_4_edge(edges, edge_index).sigmoid()
        return edges

    def process_nodes(self, x, edge_index):
        nodes = self.conv1_node(x, edge_index).relu()
        nodes_mu = self.conv_mu_node(nodes, edge_index)
        nodes_logstd = self.conv_logstd_node(nodes, edge_index)
        nodes = self.reparametrize(nodes_mu, nodes_logstd)
        nodes = self.conv5_node(nodes, edge_index).relu()
        return nodes

    def reparametrize(self, mu, logstd, divider=10):
        if self.training:
            # TODO: Consider if this multiplier is optimal
            return mu + torch.div(torch.randn_like(logstd) * torch.exp(logstd), divider)
        else:
            return mu

    def kl_loss(self, mu, logstd):
        logstd = logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def forward(self, x, edge_index):
        edges = self.process_edges(x, edge_index)
        nodes = self.process_nodes(x, edge_index)
        return edges, nodes


def train(node_optimizer, edge_optimizer, epoch):
    model.train()
    node_optimizer.zero_grad()
    edge_optimizer.zero_grad()
    node_losses = []
    edge_losses = []
    for graph in train_data:
        adj_hat, x_hat = model(graph.x.to(device), graph.edge_index.to(device))
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
def test(epoch):
    model.eval()
    node_losses = []
    edge_losses = []
    first = True
    for graph in test_data:
        adj_hat, x_hat = model(graph.x.to(device), graph.edge_index.to(device))
        if first and (logging_epochs.__contains__(epoch)):
            output_graph = train_data[0]
            output_adj_hat, output_x_hat = model(output_graph.x, output_graph.edge_index)
            visualization(output_graph, output_adj_hat, output_x_hat, epoch)
            first = False

        node_loss, edge_loss = loss_fn(adj_hat, x_hat, graph)
        node_losses.append(node_loss)
        edge_losses.append(edge_loss)
    return sum(node_losses) / len(test_data), sum(edge_losses) / len(test_data)


def visualization(graph, adj_hat, x_hat, epoch):
    # -- nodes --
    x_hat_tmp = x_hat.clone()
    diagonal = torch.diagonal(adj_hat, 0)
    x_hat_tmp = torch.where(diagonal.expand(6, x_hat.size(dim=0)).t() > 0.05, x_hat_tmp, 0)

    max_a, ids = torch.max(x_hat, 1, keepdim=True)
    x_out = torch.zeros_like(x_hat)
    x_out.scatter_(1, ids, max_a)
    x_out[x_out != 0] = 1
    x_out = torch.where(diagonal.expand(6, x_hat.size(dim=0)).t() > 0.05, x_out, 0)

    x_hat_features = process_model_dataset.plot_node_features(x_hat_tmp.detach().cpu().t().numpy(),
                                                              title="Node predictions for epoch " + str(epoch))
    writer.add_figure("Node predictions / epoch " + str(epoch), x_hat_features)
    x_out_features = process_model_dataset.plot_node_features(x_out.detach().cpu().t().numpy(),
                                                              title="Max node prediction for epoch " + str(epoch))
    writer.add_figure("Max node prediction / epoch " + str(epoch), x_out_features)

    # -- edges --
    prediction_matrix = process_model_dataset.plot_adjacency_matrix(adj_hat.detach().cpu().numpy(),
                                                                    title="Edge predictions for epoch " + str(
                                                                        epoch) + " in run " + str(
                                                                        run_name))
    writer.add_figure("Edge predictions / epoch " + str(epoch), prediction_matrix)
    output = torch.where((adj_hat > 0.05), 1, 0)

    # -- General --
    output.diagonal().fill_(0)
    bpmn = process_model_dataset.to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=graph.names_labels))
    pm4py.view_bpmn(bpmn)


def train_model(config):
    edge_optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_edges"])
    node_optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_nodes"])

    # Visulize first graph
    bpmn = process_model_dataset.to_bpmn(train_data[0])
    pm4py.view_bpmn(bpmn)

    # Log node and edge labels to tensorboard
    y_features = process_model_dataset.plot_node_features(train_data[0].y.detach().cpu().t().numpy(),
                                                          title="Node feature label")
    writer.add_figure("1-Labels/Nodes", y_features)
    label_matrix = process_model_dataset.plot_adjacency_matrix(train_data[0].adj_matrix_label,
                                                               title="Edge labels")
    writer.add_figure("1-Labels/Edges", label_matrix)

    for epoch in range(1, config["epochs"] + 1):
        node_loss, edge_loss = train(node_optimizer, edge_optimizer, epoch)
        test_node_loss, test_edge_loss = test(epoch)
        writer.add_scalar('Loss/train/node', node_loss, epoch)
        writer.add_scalar('Loss/train/edge', edge_loss, epoch)
        writer.add_scalar('Loss/test/node', test_node_loss, epoch)
        writer.add_scalar('Loss/test/edge', test_edge_loss, epoch)
        print(
            'Epoch: {:03d}, Train Node Loss: {:.7f}, Train Edge Loss: {:.7f}, Test Node Loss: {:.7f}, Test Edge Loss: {:.7f}'.format(
                epoch, node_loss, edge_loss, test_node_loss, test_edge_loss))


if __name__ == '__main__':
    run_name = "process-vgae-1"
    logging_epochs = [500, 2000, 5000, 7000]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    torch.set_printoptions(threshold=10_000, sci_mode=False)
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data-dump', 'process_vgae')
    writer = SummaryWriter(log_dir=path + '/logs/' + (time.strftime("%Y%m%d-%H%M%S") + "-" + run_name))

    dataset_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    print(dataset_path)
    process_model_dataset = ProcessModelBPMNDataset(dataset_path, transform=T.ToDevice(device))
    train_data = process_model_dataset.get_train_data(shuffle=False)
    test_data = process_model_dataset.get_test_data(shuffle=False)
    val_data = process_model_dataset.get_val_data(shuffle=False)

    model = ProcessVGAE(node_channels=6, edge_channels=process_model_dataset.max_dim).to(device)

    print("Dataset Max Dim: " + str(process_model_dataset.max_dim))
    model = model.to(device)

    train_model({"lr_edges": 0.001, "lr_nodes": 0.001, "epochs": 500})
