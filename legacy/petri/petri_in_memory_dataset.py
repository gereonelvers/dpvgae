import json
import os
import os.path as osp
import re

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx


class ProcessModelMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if self.processed_file_names[0] not in os.listdir(self.processed_dir):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['process_dataset_export_petri-net.json']

    @property
    def processed_file_names(self):
        # return [f'graph_{i}.pt' for i in range(0, 98)]
        return ["process-datasets.pt"]

    def download(self):
        # TODO: Deeper integration with dataset generator would be nice here
        # Download to `self.raw_dir`.
        # url = "https://cdn.gereonelvers.com/datasets/process-model.json"
        # path = download_url(url, self.raw_dir)
        # self._dataset = json.loads(urllib.request.urlopen(url).read())
        return

    def process(self):
        data_set = json.load(open("process_dataset_export_petri-net.json"))
        data_list = []
        for index in range(0, data_set["time_periods"]):
            # Read process-datasets from `raw_path`.
            data = Data(x=torch.FloatTensor(data_set["x"][str(index)]),
                        edge_index=torch.LongTensor(data_set["edge_index"][str(index)]).t().contiguous(),
                        edge_attr=torch.FloatTensor(data_set["edge_weights"][str(index)]),
                        names=data_set["names"][str(index)],
                        y=torch.FloatTensor(data_set["y"]["x"][str(index)]).t().contiguous(),
                        x_labels=torch.FloatTensor(data_set["y"]["x"][str(index)]),
                        edge_index_labels=torch.LongTensor(data_set["y"]["edge_index"][str(index)]).t().contiguous(),
                        names_labels=data_set["y"]["names"][str(index)])
            # process-datasets.num_nodes = len(process-datasets.x)
            # process-datasets.num_features = 1
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


def draw_graph(graph):
    # Positioning
    digit = lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    pos = {}  # np.random.rand(len(graph.x), 2)
    for i in range(len(graph.x)):
        # x = ord(graph.names[i][:1])
        # y = ord(graph.names[i][-1:])

        if graph.names[i][:2] == "T(" and graph.names[i][-1:] == ")":
            x = i
            y = i%4
        else:
            x = len(graph.names) - i
            y = 10 - i%4

        # print(digit(graph.names[i]))
        # x = digit(graph.names[i]) % 10 + i*0.1
        # y = digit(graph.names[i]) % 10 + (graph.num_nodes - i)*0.1

        # print("Generated pos: "+str(x)+" "+str(y))
        pos[i] = [x, y]

    # Name labels
    node_names = {}
    for i in range(0, len(graph.names)):
        node_names[i] = graph.names[i]

    nx.draw(to_networkx(graph), pos=pos, with_labels=True, labels=node_names)
    plt.show()


# TODO: Doesnt work
if __name__ == '__main__':
    # Dataset preparation
    dataset = ProcessModelMemoryDataset(root='./pytorch_geometric_dataset')
    dataset.download()
    dataset.process()

    # Graph drawing
    for graph in dataset:
        draw_graph(graph)
