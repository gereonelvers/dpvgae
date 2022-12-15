import os.path as osp
import pickle

import matplotlib.pyplot as plt
import pm4py
import torch
import torch_geometric
from pm4py.objects.bpmn.obj import BPMN
from torch_geometric.data import Data
from legacy.bpmn_graphvae import adj_to_edge_index
import torch.nn.functional as F

from bpmn_dataset import ProcessModelBPMNDataset


# Easily sample the adjacency matrix from past outputs.
# To use, add the following code at the bottom of the visualization method in the model:
#       filename = osp.join(pickle_path, "output_" + str(epoch) + ".pickle")
#             if not osp.exists(pickle_path):
#               os.makedirs(pickle_path)
#           with open(filename, 'wb') as f:
#               pickle.dump([adj_hat, x_out, graph.names_labels], f)
def sampling(adj_matrix, x, names, threshold: float):
    print("X: ", x)
    diagonal = torch.diagonal(adj_matrix, 0)
    x = torch.where(diagonal.expand(6, x.size(dim=0)).t() > 0.05, x,
                    0)  # Delete nodes that should not be there according to the adjacency matrix

    max_a, ids = torch.max(x, 1, keepdim=True)
    x_out = torch.zeros_like(x)
    x_out.scatter_(1, ids, max_a)
    x_out[x_out != 0] = 1
    print("X_out: ", x_out)

    output = torch.where((adj_matrix > threshold), 1, 0)
    output.diagonal().fill_(0)
    return to_bpmn(Data(x=x_out, edge_index=adj_to_edge_index(output), names=names))


def plot_matrix(matrix):
    plt.imshow(matrix)
    # Add the values to the plot using the annotate function
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Not sure why this is flipped, but it works ¯\_(ツ)_/¯
            plt.annotate(str(round(matrix[j, i].item(), ndigits=4)), (i, j), ha='center', va='center', fontsize=5)
    plt.show()


def to_bpmn(graph):
    nodes = []
    for i in range(0, len(graph.x)):
        if torch.eq(graph.x[i], torch.tensor([1, 0, 0, 0, 0, 0])).all():
            nodes.append(BPMN.StartEvent(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 1, 0, 0, 0, 0])).all():
            nodes.append(BPMN.EndEvent(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 0, 1, 0, 0, 0])).all():
            nodes.append(BPMN.Task(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 1, 0, 0])).all():
            nodes.append(BPMN.ExclusiveGateway(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 1, 0])).all():
            nodes.append(BPMN.ParallelGateway(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 0, 1])).all():
            nodes.append(BPMN.InclusiveGateway(name=graph.names[i]))
        elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 0, 0])).all():
            # This is a stupid fix
            nodes.append(BPMN.BPMNNode(name=graph.names[i]))
        else:
            raise Exception("Unknown node type, tensor:" + str(graph.x[i]))
    edges = []
    # edge_index is in COO format
    for i in range(0, len(graph.edge_index.t())):
        source_node = graph.edge_index.t()[i][0]
        target_node = graph.edge_index.t()[i][1]
        source = nodes[source_node]
        target = nodes[target_node]
        edges.append(BPMN.Flow(source=source, target=target))

    # Remove nodes where adjacency matrix diagonal is 0
    diagonal = torch.diagonal(create_adjacency_matrix(graph.edge_index, len(graph.x), 15))  # TODO: Hardcoded
    diagonal.tolist()
    for i in range(0, len(diagonal)):
        if diagonal[i] == 0:
            nodes[i] = BPMN.BPMNNode(name=graph.names[i])
    print("Created BPMN with " + str(len(nodes)) + " nodes and " + str(len(edges)) + " edges")
    return BPMN(nodes=list(filter(lambda a: type(a) != BPMN.BPMNNode, nodes)), flows=edges)


def create_adjacency_matrix(edge_index, num_nodes, max_dim):
    # Create adjacency matrix
    matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(edge_index=edge_index,
                                                                  num_nodes=num_nodes).todense()
    matrix = torch.from_numpy(matrix)
    matrix.fill_diagonal_(1)
    matrix = F.pad(matrix, pad=(0, max_dim - num_nodes, 0, max_dim - num_nodes), mode='constant', value=0)
    matrix = torch.clamp(matrix, 0, 1)  # TODO: Sometimes values above 1 are produced here. Why?
    return matrix


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data-dump', 'dpvgae', "pickles",
                    "<model_name>.pkl")
    filename = osp.join(path, "output_" + str(2000) + ".pickle")
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    sampling(file[0], file[1], file[2], 0.03)
