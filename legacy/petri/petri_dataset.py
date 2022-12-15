import json
import os.path as osp
import random

import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt, pyplot
from torch_geometric.data import Dataset, download_url, Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_networkx


class Graph(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, names=None, x_labels=None, edge_index_labels=None, names_labels=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos, **kwargs)
        self.names = names
        self.x_labels = x_labels
        self.edge_index_labels = edge_index_labels
        self.names_labels = names_labels

# Might be necessary: https://github.com/pyg-team/pytorch_geometric/issues/4588
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


class ProcessModelPetriDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.download()
        self.process()


    @property
    def raw_file_names(self):
        return ['process_dataset_export_petri-net.json']

    @property
    def processed_file_names(self):
        # TODO: This really shouldn't be hardcoded
        return [f'graph_{i}.pt' for i in range(0, 98)]

    def download(self):
        # TODO: Deeper integration with dataset generator would be nice here
        # Download to `self.raw_dir`.
        # url = "https://cdn.gereonelvers.com/datasets/process-model.json"
        # path = download_url(url, self.raw_dir)
        # self._dataset = json.loads(urllib.request.urlopen(url).read())
        return

    def process(self):
        data_set = json.load(open("process_dataset_export_petri-net.json"))
        for index in range(0, data_set["time_periods"]):
            # Read process-datasets from `raw_path`.
            data = Graph(x=torch.FloatTensor(data_set["x"][str(index)][:25]),
                         edge_index=torch.clamp(torch.LongTensor(data_set["edge_index"][str(index)]).t().contiguous(), min=0, max=24),
                         edge_attr=torch.FloatTensor(data_set["edge_weights"][str(index)]),
                         names=data_set["names"][str(index)][:25],
                         y=torch.FloatTensor(data_set["y"]["x"][str(index)]).t().contiguous()[:25],
                         x_labels=torch.FloatTensor(data_set["y"]["x"][str(index)][:25]),
                         edge_index_labels=torch.clamp(torch.LongTensor(data_set["y"]["edge_index"][str(index)]).t().contiguous(), min=0, max=24),
                         names_labels=data_set["y"]["names"][str(index)][:25],
                         edge_index_tmp=torch.clamp(
                             torch.FloatTensor(data_set["edge_index"][str(index)]).t().contiguous(), min=0,
                             max=24),
                         adj_matrix=self.create_adjacency_matrix(edge_index=torch.clamp(torch.LongTensor(data_set["edge_index"][str(index)]).t().contiguous()[0:2, :25], min=0, max=24), num_nodes=25),
                         adj_matrix_label=self.create_adjacency_matrix(edge_index=torch.clamp(torch.LongTensor(data_set["y"]["edge_index"][str(index)]).t().contiguous()[0:2, :25], min=0, max=24), num_nodes=25)
                         )
            # process-datasets.num_nodes = len(process-datasets.x)
            # process-datasets.num_features = 1
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'graph_{index}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return data

    def get_train_data(self, shuffle=True):
        data = []
        for i in range(0, int(self.len() * 0.7)):
            data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def get_test_data(self, shuffle=True):
        data = []
        for i in range(int(self.len() * 0.7), int(self.len() * 0.9)):
            data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def get_val_data(self, shuffle=True):
        data = []
        for i in range(int(self.len() * 0.9), self.len()):
            data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def num_node_features(self) -> int:
        return 1

    def num_edge_features(self) -> int:
        return 1

    def num_node_classes(self) -> int:
        return 1

    def draw_graph(self, graph, title=None):
        # Positioning
        pos = {}  # np.random.rand(len(graph.x), 2)
        for i in range(len(graph.x)):
            # x = ord(graph.names[i][:1])
            # y = ord(graph.names[i][-1:])

            if graph.names[i][:2] == "T(" and graph.names[i][-1:] == ")":
                x = i
                y = i % 4
            else:
                x = len(graph.names) - i
                y = 10 - i % 4

            # print(digit(graph.names[i]))
            # x = digit(graph.names[i]) % 10 + i*0.1
            # y = digit(graph.names[i]) % 10 + (graph.num_nodes - i)*0.1

            # print("Generated pos: "+str(x)+" "+str(y))
            pos[i] = [x, y]

        # Name labels
        node_names = {}
        for i in range(0, len(graph.names)):
            node_names[i] = graph.names[i]
        figure = plt.figure()
        nx.draw(to_networkx(graph), pos=pos, with_labels=True, labels=node_names, label="title")

        # if(graph.x_labels is not None):
        #     # print("Data object contains:")
        #     # print("x: "+str(graph.x))
        #     # print("x_labels: "+str(graph.x_labels))
        #     # print("edge_index: "+str(graph.edge_index))
        #     # print("edge_index_labels: "+str(graph.edge_index_labels))
        #     # print("names: "+str(graph.names))
        #     # print("names_labels: "+str(graph.names_labels))
        #     # Positioning
        #     pos = {}  # np.random.rand(len(graph.x), 2)
        #     for i in range(len(graph.x_labels)):
        #         # x = ord(graph.names[i][:1])
        #         # y = ord(graph.names[i][-1:])
        #         if graph.names_labels[i][:2] == "T(" and graph.names_labels[i][-1:] == ")":
        #             x = i
        #             y = i % 4
        #         else:
        #             x = len(graph.names) - i
        #             y = 10 - i % 4
        #
        #         # print(digit(graph.names[i]))
        #         # x = digit(graph.names[i]) % 10 + i*0.1
        #         # y = digit(graph.names[i]) % 10 + (graph.num_nodes - i)*0.1
        #
        #         # print("Generated pos: "+str(x)+" "+str(y))
        #         pos[i] = [x, y]
        #     node_names = {}
        #     for i in range(0, len(graph.names_labels)):
        #         node_names[i] = graph.names[i]
        #     nx.draw(to_networkx(Graph(x=graph.x_labels, edge_index=graph.edge_index_labels)), pos=pos, with_labels=True,
        #             labels=node_names)
        return figure

    def create_adjacency_matrix(self, edge_index, num_nodes):
        # Create adjacency matrix
        matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=25).todense()
        matrix = torch.from_numpy(matrix)
        matrix.fill_diagonal_(1)  # TODO: This should only fill for nodes that actually exist (not always 25!)
        matrix = torch.clamp(matrix, 0, 1)  # TODO: Sometimes values above 1 are produced here. Why?
        return matrix

    def plot_adjacency_matrix(self, matrix):
        # Plot adjacency matrix
        pyplot.figure(figsize=(25, 25))
        pyplot.imshow(matrix)
        pyplot.show()


if __name__ == '__main__':
    # Dataset preparation
    dataset = ProcessModelPetriDataset(root='./pytorch_geometric_dataset')
    dataset.download()
    dataset.process()
    # print(dataset.get(40).adj_matrix)
    pyplot.figure(figsize=(25, 25))
    pyplot.imshow(dataset.get(40).adj_matrix)
    pyplot.show()
    edge_index, edge_attributes = torch_geometric.utils.dense_to_sparse(dataset.get(40).adj_matrix)
    print(dataset.get(40).edge_index)
    print(edge_index)

    # # Graph drawing
    # for graph in dataset:
    #     dataset.draw_graph(graph)
