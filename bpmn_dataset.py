import json
import os
import os.path as osp
import random

import pm4py
import torch
import torch_geometric
from matplotlib import pyplot as pyplot
from pm4py.objects.bpmn.obj import BPMN
from tensorboardX import SummaryWriter
from torch_geometric.data import Dataset, download_url, Data
import pm4py.objects.bpmn.obj as bpmn_obj
import torch.nn.functional as F


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


class ProcessModelBPMNDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.download()
        self.process()

    @property
    def raw_file_names(self):
        return ['process_dataset_export_bpmn.json']

    @property
    def processed_file_names(self):
        # TODO: This needs to be set manually for now
        return [f'graph_{i}.pt' for i in range(0, 9)]

    def download(self):
        # Deeper integration with dataset generator would be nice here
        # Download to `self.raw_dir`.
        # url = "https://cdn.gereonelvers.com/datasets/process-model.json"
        # path = download_url(url, self.raw_dir)
        # self._dataset = json.loads(urllib.request.urlopen(url).read())
        return

    def process(self):
        print(os.path.dirname(os.path.realpath(__file__)))
        data_set = json.load(
            open(osp.join(os.path.dirname(os.path.realpath(__file__)), "process_dataset_export_bpmn.json")))
        self.max_dim = data_set["max_dim"]
        for index in range(0, data_set["time_periods"]):
            # Read process-datasets from `raw_path`.
            data = Data(x=F.pad(torch.FloatTensor(data_set["x"][str(index)]),
                                pad=(0, 0, 0, self.max_dim - len(data_set["x"][str(index)])), mode='constant',
                                value=0),
                        edge_index=torch.LongTensor(data_set["edge_index"][str(index)]).t().contiguous(),
                        names=data_set["names"][str(index)] + [""] * (
                                self.max_dim - len(data_set["names"][str(index)])),
                        y=F.pad(torch.FloatTensor(data_set["y"]["x"][str(index)]),
                                pad=(0, 0, 0, self.max_dim - len(data_set["y"]["x"][str(index)])), mode='constant',
                                value=random.uniform(0, 1)),
                        x_labels=F.pad(torch.FloatTensor(data_set["y"]["x"][str(index)]),
                                       pad=(0, 0, 0, self.max_dim - len(data_set["y"]["x"][str(index)])),
                                       mode='constant',
                                       value=0),
                        edge_index_labels=torch.LongTensor(data_set["y"]["edge_index"][str(index)]).t().contiguous(),
                        names_labels=data_set["y"]["names"][str(index)] + [""] * (
                                self.max_dim - len(data_set["y"]["names"][str(index)])),
                        edge_index_tmp=torch.FloatTensor(data_set["edge_index"][str(index)]).t().contiguous(),
                        adj_matrix=self.create_adjacency_matrix(
                            edge_index=torch.LongTensor(data_set["edge_index"][str(index)]).t().contiguous(),
                            num_nodes=len(data_set["x"][str(index)]), max_dim=self.max_dim),
                        adj_matrix_label=self.create_adjacency_matrix(
                            edge_index=torch.LongTensor(data_set["y"]["edge_index"][str(index)]).t().contiguous(),
                            num_nodes=len(data_set["y"]["x"][str(index)]), max_dim=self.max_dim)
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

    def get_train_data(self, shuffle=True, time_sorted=True):
        data = []
        if time_sorted:
            for i in range(0, int(self.len() * 0.5)):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        else:
            for i in range(0, int(self.len() * 0.5)):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{random.randint(0, self.len() - 1)}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def get_test_data(self, shuffle=True, time_sorted=True):
        data = []
        if time_sorted:
            for i in range(int(self.len() * 0.5), int(self.len() * 0.8)):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        else:
            for i in range(int(self.len() * 0.5), int(self.len() * 0.8)):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{random.randint(0, self.len() - 1)}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def get_val_data(self, shuffle=True, time_sorted=True):
        data = []
        if time_sorted:
            for i in range(int(self.len() * 0.8), self.len()):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{i}.pt')))
        else:
            for i in range(int(self.len() * 0.8), self.len()):
                data.append(torch.load(osp.join(self.processed_dir, f'graph_{random.randint(0, self.len() - 1)}.pt')))
        if shuffle:
            random.shuffle(data)
        return tuple(data)

    def num_node_features(self) -> int:
        return 6

    def num_edge_features(self) -> int:
        return 0

    def num_node_classes(self) -> int:
        return 6

    def create_adjacency_matrix(self, edge_index, num_nodes, max_dim):
        # Create adjacency matrix
        matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(edge_index=edge_index,
                                                                      num_nodes=num_nodes).todense()
        matrix = torch.from_numpy(matrix)
        matrix.fill_diagonal_(1)
        matrix = F.pad(matrix, pad=(0, max_dim - num_nodes, 0, max_dim - num_nodes), mode='constant', value=0)

        matrix = torch.clamp(matrix, 0, 1)  # TODO: Sometimes values above 1 are produced here. Why?
        return matrix

    def plot_adjacency_matrix(self, matrix, title=None):
        # Plot adjacency matrix
        figure = pyplot.figure(figsize=(matrix.shape[0], matrix.shape[1]))
        pyplot.title(label=title, fontsize=44)
        pyplot.imshow(matrix)
        return figure

    def plot_node_features(self, x, title=None):
        # Plot node features
        figure = pyplot.figure(figsize=(x.shape[1], x.shape[0]))
        pyplot.title(label=title, fontsize=44)
        pyplot.imshow(x)
        return figure

    def to_bpmn(self, graph):
        nodes = []
        for i in range(0, len(graph.x)):
            if torch.eq(graph.x[i], torch.tensor([1, 0, 0, 0, 0, 0])).all():
                nodes.append(bpmn_obj.BPMN.StartEvent(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 1, 0, 0, 0, 0])).all():
                nodes.append(bpmn_obj.BPMN.EndEvent(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 0, 1, 0, 0, 0])).all():
                nodes.append(bpmn_obj.BPMN.Task(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 1, 0, 0])).all():
                nodes.append(bpmn_obj.BPMN.ExclusiveGateway(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 1, 0])).all():
                nodes.append(bpmn_obj.BPMN.ParallelGateway(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 0, 1])).all():
                nodes.append(bpmn_obj.BPMN.InclusiveGateway(name=graph.names[i]))
            elif torch.eq(graph.x[i], torch.tensor([0, 0, 0, 0, 0, 0])).all():
                # TODO: This is a stupid fix
                nodes.append(bpmn_obj.BPMN.BPMNNode(name=graph.names[i]))
            else:
                raise Exception("Unknown node type, tensor:" + str(graph.x[i]))
        edges = []
        # edge_index is in COO format
        for i in range(0, len(graph.edge_index.t())):
            source_node = graph.edge_index.t()[i][0]
            target_node = graph.edge_index.t()[i][1]
            source = nodes[source_node]
            target = nodes[target_node]
            edges.append(bpmn_obj.BPMN.Flow(source=source, target=target))

        # Remove nodes where adjacency matrix diagonal is 0
        diagonal = torch.diagonal(self.create_adjacency_matrix(graph.edge_index, len(graph.x), self.max_dim))
        diagonal.tolist()
        for i in range(0, len(diagonal)):
            if diagonal[i] == 0:
                nodes[i] = bpmn_obj.BPMN.BPMNNode(name=graph.names[i])
        print("Created BPMN with " + str(len(nodes)) + " nodes and " + str(len(edges)) + " edges")

        return bpmn_obj.BPMN(nodes=list(filter(lambda a: type(a) != bpmn_obj.BPMN.BPMNNode, nodes)), flows=edges)


if __name__ == '__main__':
    # This works as a manual override to trigger recreation from current JSON
    dataset = ProcessModelBPMNDataset(root='./data-dump/dpvgae')
    dataset.download()
    dataset.process()
