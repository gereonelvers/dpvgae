import os.path as osp

import pm4py
import torch

import torch_geometric.transforms as T
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import time

import argparse
from bpmn_dataset import ProcessModelBPMNDataset
from layers import DirectedProcessVGAE, DirectedEdgeEncoder
from legacy.bpmn_graphvae import adj_to_edge_index


def loss_fn(adjacency_matrix, x_hat, graph, epoch):
    # Edge loss: Weighted BCE + KL
    # loss_function = BCELoss()
    weight = torch.zeros_like(adjacency_matrix)
    weight[graph.adj_matrix_label == 0] = 0.2
    weight[graph.adj_matrix_label == 1] = 1
    adjacency_matrix = torch.where(adjacency_matrix < 0.1, adjacency_matrix/2, adjacency_matrix)
    loss_function = BCEWithLogitsLoss(weight=weight)
    edge_loss = loss_function(adjacency_matrix, graph.adj_matrix_label.to(device))
    kl_loss_edge = model.kl_loss(adjacency_matrix, graph.adj_matrix_label.to(device))
    edge_loss += (1 / (10*process_model_dataset.max_dim)) * kl_loss_edge
    # Node loss: CE with node embeddings being one-hot encodings for node type
    node_loss_function = CrossEntropyLoss()
    x_hat_tmp = x_hat.clone()  # Sometimes needed on CUDA to prevent modification exceptions for some reason
    y = graph.y.clone().to(device)  # Same lol
    argmax = torch.argmax(y, dim=1)  # CE loss requires feature labels as indexes, not one-hot
    node_loss = node_loss_function(x_hat_tmp, argmax.to(device))
    kl_loss_node = model.kl_loss(x_hat_tmp, y.to(device))
    node_loss += (1 / process_model_dataset.max_dim) * kl_loss_node
    return edge_loss, node_loss


def train(edge_optimizer, node_optimizer, epoch):
    model.train()
    edge_optimizer.zero_grad()
    edge_losses = []
    node_losses = []
    for graph in train_data:
        adj_hat, x_hat = model(graph.to(device))
        edge_loss, node_loss = loss_fn(adj_hat, x_hat, graph, epoch)
        edge_losses.append(edge_loss)
        node_losses.append(node_loss)
    total_edge_loss, total_node_loss = sum(edge_losses), sum(node_losses)
    total_edge_loss.backward()
    total_node_loss.backward()
    edge_optimizer.step()
    node_optimizer.step()
    return float(total_edge_loss) / len(train_data), float(total_node_loss) / len(train_data)


@torch.no_grad()
def test(epoch):
    model.eval()
    edge_losses = []
    node_losses = []
    first = True
    for graph in test_data:
        adj_hat, x_hat = model(graph)
        if first and (logging_epochs.__contains__(epoch)):
            output_graph = train_data[0]
            output_adj_hat, output_x_hat = model(output_graph)
            visualization(output_graph, output_adj_hat, output_x_hat, epoch)
            first = False
        edge_loss, node_loss = loss_fn(adj_hat, x_hat, graph, epoch)
        edge_losses.append(edge_loss)
        node_losses.append(node_loss)
    return sum(edge_losses) / len(test_data), sum(node_losses) / len(test_data)


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
    output = torch.where((adj_hat > 0.1), 1, 0)

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
        edge_loss, node_loss = train(edge_optimizer, node_optimizer, epoch)
        test_edge_loss, test_node_loss = test(epoch)
        writer.add_scalar('Loss/train/edge', edge_loss, epoch)
        writer.add_scalar('Loss/test/edge', test_edge_loss, epoch)
        writer.add_scalar('Loss/train/node', node_loss, epoch)
        writer.add_scalar('Loss/test/node', test_node_loss, epoch)
        print('Epoch: {:03d}, Train Edge: {:.7f}, Test Edge: {:.7f}, Train Node: {:.7f}, Test Node: {:.7f}'.format(
            epoch, edge_loss, test_edge_loss, node_loss, test_node_loss))


if __name__ == '__main__':
    run_name = "dpvgae-bpi-2017-100-steps"
    logging_epochs = [2000, 5000, 7000]
    edge_encoder_hidden_channels = 32
    lr_edges = 0.001
    lr_nodes = 0.001
    epochs = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    torch.set_printoptions(threshold=10_000, sci_mode=False)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data-dump', 'dpvgae')
    writer = SummaryWriter(log_dir=path + '/logs/' + (time.strftime("%Y%m%d-%H%M%S") + "-" + run_name))

    process_model_dataset = ProcessModelBPMNDataset(path, transform=T.ToDevice(device))
    train_data = process_model_dataset.get_train_data()
    test_data = process_model_dataset.get_test_data()
    val_data = process_model_dataset.get_val_data()

    model = DirectedProcessVGAE(
        edge_encoder=DirectedEdgeEncoder(in_channels=6, out_channels=process_model_dataset.max_dim,
                                         hidden_channels=edge_encoder_hidden_channels),
    ).to(device)

    print("Dataset Max Dim: " + str(process_model_dataset.max_dim))
    model = model.to(device)

    train_model({"lr_edges": lr_edges, "lr_nodes": lr_nodes, "epochs": epochs})
    torch.save(model.state_dict(), 'dpvgae.pth')
