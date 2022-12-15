import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, GATConv, Linear, GCNConv, ChebConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import reset, glorot, constant
from torch_sparse import SparseTensor


class DirectedInnerProductDecoder(torch.nn.Module):
    @staticmethod
    def forward_all(s, t, sigmoid=False):
        adj = torch.matmul(s, t.t())
        return torch.sigmoid(adj) if sigmoid else adj


class DirectedGCNConv(MessagePassing):
    # Just here to suppress warnings :)
    def edge_update(self) -> Tensor:
        raise NotImplementedError
    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def __init__(self, in_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True, adaptive=False):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.alpha = alpha
        self.beta = beta
        self.self_loops = self_loops
        self.adaptive = adaptive

    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        in_degree = degree(col)
        out_degree = degree(row)
        alpha = self.alpha
        beta = self.beta
        in_norm_inv = pow(in_degree, -alpha)
        out_norm_inv = pow(out_degree, -beta)
        in_norm = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm = in_norm * out_norm
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class EdgeSourceEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True,
                 adaptive=False):
        super(EdgeSourceEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        # self.conv2 = DirectedGCNConv(hidden_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        # self.conv3 = DirectedGCNConv(hidden_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv4 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        # x = self.conv1(x, edge_index)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, torch.flip(edge_index, [0]))
        return x


class EdgeTargetEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True,
                 adaptive=False):
        super(EdgeTargetEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        # self.conv2 = DirectedGCNConv(hidden_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        # self.conv3 = DirectedGCNConv(hidden_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv4 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, x, edge_index):
        x = self.conv1(x, torch.flip(edge_index, [0])).relu()
        # x = self.conv1(x, torch.flip(edge_index, [0]))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        return x


class DirectedEdgeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True,
                 adaptive=False):
        super(DirectedEdgeEncoder, self).__init__()
        self.source_conv_mu = EdgeSourceEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops,
                                                adaptive)
        self.source_conv_logstd = EdgeSourceEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops,
                                                    adaptive)
        self.target_conv_mu = EdgeTargetEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops,
                                                adaptive)
        self.target_conv_logstd = EdgeTargetEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops,
                                                    adaptive)

    def forward(self, s, t, edge_index):
        s_mu = self.source_conv_mu(s, edge_index)
        s_logstd = self.source_conv_logstd(s, edge_index)
        t_mu = self.target_conv_mu(t, edge_index)
        t_logstd = self.target_conv_logstd(t, edge_index)
        s = self.reparametrize_edges(s_mu, s_logstd)
        t = self.reparametrize_edges(t_mu, t_logstd)
        return s, t

    def reparametrize_edges(self, mu, logstd, divider=5):
        if self.training:
            # TODO: Consider if this multiplier is optimal
            return mu + torch.div(torch.randn_like(logstd) * torch.exp(logstd), divider)
        else:
            return mu


class NodeModel(torch.nn.Module):
    def __init__(self, node_channels):
        super(NodeModel, self).__init__()
        self.conv1_node = GCNConv(node_channels, node_channels)
        self.conv2_node = GCNConv(node_channels, node_channels)
        self.conv3_node = GCNConv(node_channels, node_channels)
        self.conv4_node = GCNConv(node_channels, node_channels)
        self.conv_mu_node = GCNConv(node_channels, node_channels)
        self.conv_logstd_node = GCNConv(node_channels, node_channels)
        self.conv5_node = GCNConv(node_channels, node_channels)
        self.conv6_node = GCNConv(node_channels, node_channels)
        self.conv7_node = GCNConv(node_channels, node_channels)
        self.conv8_node = GCNConv(node_channels, node_channels)

    # TODO: This *works* but probably isn't optimal
    def forward(self, x, edge_index):
        nodes = self.conv1_node(x, edge_index).relu()
        nodes = self.conv2_node(nodes, edge_index).relu()
        #nodes = self.conv3_node(nodes, edge_index).relu()
        #nodes = self.conv4_node(nodes, edge_index).relu()
        nodes_mu = self.conv_mu_node(nodes, edge_index)
        nodes_logstd = self.conv_logstd_node(nodes, edge_index)
        nodes = self.reparametrize_nodes(nodes_mu, nodes_logstd)
        nodes = self.conv5_node(nodes, edge_index).relu()
        nodes = self.conv6_node(nodes, edge_index).relu()
        #nodes = self.conv7_node(nodes, edge_index).relu()
        #nodes = self.conv8_node(nodes, edge_index).relu()
        return nodes

    def reparametrize_nodes(self, mu, logstd, divider=10):
        if self.training:
            # This also *works* but probably isn't optimal
            return mu + torch.div(torch.randn_like(logstd) * torch.exp(logstd), divider)
        else:
            return mu


class DirectedProcessVGAE(torch.nn.Module):
    def __init__(self, edge_encoder, edge_decoder=DirectedInnerProductDecoder(), node_model=NodeModel(6)):
        super(DirectedProcessVGAE, self).__init__()
        self.edge_encoder = edge_encoder
        self.edge_decoder = edge_decoder
        self.node_model = node_model
        DirectedProcessVGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.edge_decoder)
        reset(self.node_model)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        s, t = self.edge_encoder(x, x, edge_index)
        adj_pred = self.edge_decoder.forward_all(s, t)
        x = self.node_model(x, edge_index)
        return adj_pred, x

    def kl_loss(self, mu, logstd):
        logstd = logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
