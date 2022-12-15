import json
import warnings

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from torch_geometric_temporal import DynamicGraphTemporalSignal, ChickenpoxDatasetLoader, TwitterTennisDatasetLoader, \
    EnglandCovidDatasetLoader, temporal_signal_split, WikiMathsDatasetLoader, EvolveGCNO
from six.moves import urllib

# from GRAN.model.gran_mixture_bernoulli import GRANMixtureBernoulli

from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm


# TODO: Documentation details :)
# TODO: Fix json parsing form copy-pasted to fitting
class ProcessModelDriftDataset(object):
    """
    Dataset consisting of process models mined on a randomly generated process model to showcase
    and work with concept drift over time
    """

    def __init__(self):
        self._read_web_data()

    """
    Expected json layout:
    {
        "time_periods": <n>,        # No. of time-step process models
        "edge_index": {             # For each time step t (0..n), a list of all edges
            <t>: [
                [<n1>,<n2>],
                ...
            ],
        },
        "edge_weights": {             # For each time step t (0..n), a list of all edges
            <t>: [
                <w1>,
                ...
            ],
        },
        "x": {                      # For each time step t (0..n), a list of all node types (as one-hot encoding)
            <t>: [<n1>,...]
        },
        "y": {                      # For each time step t (0..n), labels to train against, in the same format as above
            "edge_index": {         # (Currently, these are just the nodes and edges of the temporally next graph)
                <t>: [
                    [<n1>,<n2>],
                    ...
                ],
            },
            "x": {
                <t>: [<n1>,...]
            },
        }
    }
    """

    def _read_web_data(self):
        url = "https://cdn.gereonelvers.com/datasets/process-model.json"
        # self._dataset = json.loads(urllib.request.urlopen(url).read())
        self._dataset = json.load(open("../process_dataset_export_petri-net.json"))

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_index"][str(time)]).T
            )

    # This is just all 0s for now
    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_weights"][str(time)])
            )

    def _get_features(self):
        self._features = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._features.append(
                np.array(self._dataset["x"][str(time)])
            )

    def _get_targets(self):
        self._targets = []
        for time in range(self._dataset["time_periods"] - self.lags):
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            self._targets.append(
                # np.array([self._dataset["y"]["edge_index"][str(time)], self._dataset["y"]["x"][str(time)][:27]])
                np.array(self._dataset["y"]["x"][str(time)])[:27]
            )
        # self._targets = np.array([self._dataset["y"]])

    # TODO: Figure out what to do about lag var here
    def get_dataset(self, lags: int = 1) -> DynamicGraphTemporalSignal:
        """Returning the process model drift process-datasets iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - The process model drift dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self._features, self._targets
        )
        return dataset


# Simple model to use as placeholder to test dataset
# TODO: Remove/Build out into separate module once dataset confirmed working
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCN, self).__init__()
        self.linear = torch.nn.Linear(out_features, in_features)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(x[:27])
        h = self.linear(h)
        return h


# class PyGTemporalExperimentalNetwork(torch.nn.Module):
#     def __init__(self):
#         super(PyGTemporalExperimentalNetwork, self).__init__()
#         self.evolveGCN = EvolveGCNO(27)
#         self.linear = torch.nn.Linear(27, 2)
#
#     def forward(self, x, edge_index):
#         h = self.evolveGCN(x, edge_index)
#         h = F.relu(h)
#         h = self.linear(h)
#         return h


if __name__ == '__main__':
    # Reference/Utility printouts
    print("Working with torch " + torch.__version__)
    # ChickenpoxDatasetLoader  # StaticGraphTemporalSignal
    # WikiMathsDatasetLoader
    # TwitterTennisDatasetLoader  # DynamicGraphTemporalSignal, but with weird one-hot encoding
    # EnglandCovidDatasetLoader  # DynamicGraphTemporalSignal

    loader = ProcessModelDriftDataset()
    data_set = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(data_set, train_ratio=0.7)
    model = SimpleGCN(27, 27)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # TODO: Adjust primitive loss computation to match dimensionality
    model.train()
    for epoch in tqdm(range(50)):
        for time, snapshot in enumerate(train_dataset):
            # print(snapshot.x)
            # print(snapshot.edge_index)
            # print(snapshot.edge_attr)
            # print(str(snapshot.y))
            # nx.draw(to_networkx(snapshot))
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = torch.mean((y_hat - snapshot.y[:27]) ** 2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))
