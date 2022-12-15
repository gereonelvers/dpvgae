import argparse
import os.path as osp

import torch
import torch_geometric

import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

from torch_geometric.data import Data

from petri_dataset import ProcessModelPetriDataset
from petri_in_memory_dataset import ProcessModelMemoryDataset


class ProcessModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, adj_matrix, x):
        return torch.clamp(self.linear_2(self.relu(self.linear(adj_matrix))), min=0, max=1), torch.clamp(
            self.linear_2(self.relu(self.linear(x))), min=0, max=1)


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
    total_loss = 0
    for graph in train_data:
        model.train()
        optimizer.zero_grad()
        # z_x, z_edge_index = model.forward(graph.x, graph.edge_index_tmp)
        output_matrix, output_x = model.forward(graph.adj_matrix.to(device), graph.x.to(device))
        # TODO: Adapt loss function:
        # Step 1:
        # - Validate formal correctness of petri net
        # - Make sure weighting of loss function components is correct
        # Step 2:
        # - Process Redesign KPI stuff +
        # - Process Model Complexity stuff

        loss = torch.nn.functional.mse_loss(output_matrix,
                                            graph.adj_matrix_label.to(device)) + torch.nn.functional.mse_loss(
            output_x, graph.y.to(device))

        # loss = loss + (1 / graph.num_nodes) * model.kl_loss()
        loss.backward()
        total_loss += float(loss)
        optimizer.step()
    return total_loss / len(train_data)


@torch.no_grad()
def test(data):
    loss = 0
    first = True
    for graph in data:
        model.eval()
        output_matrix, output_x = model.forward(graph.adj_matrix.to(device), graph.x.to(device))

        if first:
            first = False
            # Log graph to tensorboard
            original_graph = plt.figure(figsize=(25, 25))
            plt.imshow(graph.adj_matrix_label)
            writer.add_figure("{}/Adjacency matrix (label)".format(epoch), original_graph)

            generated_graph = plt.figure(figsize=(25, 25))
            plt.imshow(output_matrix.cpu())
            writer.add_figure("{}/Adjacency matrix (output)".format(epoch), generated_graph)

            # Convert output back to graph, visualize that
            # edge_index, edge_attributes = torch_geometric.utils.dense_to_sparse(output_adj)
            # process_model_dataset.draw_graph(graph)
            writer.add_figure("{}/Petri Net (label)".format(epoch), process_model_dataset.draw_graph(
                Data(x=graph.y, edge_index=graph.edge_index_labels, names=graph.names_labels, x_labels=None)))

            edge_index, edge_attributes = torch_geometric.utils.dense_to_sparse(output_matrix)
            writer.add_figure("{}/Petri Net (output)".format(epoch), process_model_dataset.draw_graph(
                Data(x=output_x, edge_index=edge_index, names=graph.names_labels, x_labels=None)))

        loss += torch.nn.functional.mse_loss(output_matrix,
                                             graph.adj_matrix_label.to(device)) + torch.nn.functional.mse_loss(
            output_x, graph.y.to(device))
    return loss / len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # TODO: uncomment once working
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data-dump/petri_adjacency_cnn')
    writer = SummaryWriter(log_dir=path + '/logs/' + time.strftime("%Y%m%d-%H%M%S"))

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    in_memory_dataset = ProcessModelMemoryDataset(path, transform=transform)
    process_model_dataset = ProcessModelPetriDataset(path, transform=transform)

    train_data = process_model_dataset.get_train_data(shuffle=False)
    test_data = process_model_dataset.get_test_data(shuffle=False)
    val_data = process_model_dataset.get_val_data(shuffle=False)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])
    process_model_dataset = ProcessModelPetriDataset(path, transform=transform)

    in_channels, out_channels = 25, 25

    torch.set_printoptions(threshold=10_000)

    model = ProcessModel(in_channels, out_channels)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        loss = train()
        test_loss = test(test_data)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.8f}, Test Loss: {test_loss:.8f}')

    # Idea:
    # - Generate fixed size (=max size) inputs and outputs
    # - Loss is reconstruction of probabilistric adjacency matrix (with the diagonal being the probability of the node existing), plus maybe other reconstruction loss stuff from autoencoder
    # - Added to loss: Reconstruction loss on node feature matrix (maybe need to maintain that as a separate matrix, might be able to put it in the adjacency matrix)

    # TODO: Generate new graphs after training to test generation capabilities
    # # Generate testing output
    # model.eval()
    # graph = process_model_dataset.get(2)
    # output_adj, output_x = model.forward(graph.adj_matrix, graph.x)
    #
    # # Visualize adjacency matrix
    # process_model_dataset.plot_adjacency_matrix(graph.adj_matrix)
    # process_model_dataset.plot_adjacency_matrix(torch.round(output_adj).detach())
    #
    # # Convert output back to graph, visualize that
    # edge_index, edge_attributes = torch_geometric.utils.dense_to_sparse(output_adj)
    # process_model_dataset.draw_graph(graph)
    # process_model_dataset.draw_graph(Data(x=output_x, edge_index=edge_index, names=graph.names, x_labels=None))
    # plt.show()
    # # petri_net = pm4py.objects.conversion.process_tree.converter. petri_to_bpmn(to_networkx(graph))
