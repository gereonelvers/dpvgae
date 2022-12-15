import json

import numpy as np
import pm4py
import warnings

import torch
from conceptdrift.drifts.gradual import generate_log_with_gradual_drift
from conceptdrift.drifts.sudden import generate_log_with_sudden_drift
from conceptdrift.drifts.recurring import generate_log_with_recurring_drift
from conceptdrift.drifts.incremental import generate_log_with_incremental_drift
from conceptdrift.source.process_tree_controller import generate_specific_trees
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.visualization.bpmn import visualizer
from torch_geometric.data import Data
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

# TODO: Outsource parameters to config file
from typing import List


class ProcessDatasetGenerator:
    @staticmethod
    def mine_process_models(log, time_steps) -> List[ProcessTree]:
        """
        Splits process log into separate time steps and mines process models on each
        :param log: event log to be mined
        :param time_steps: number of separate time steps the log should be split into
        :return: list[ProcessTree] of mined process models
        """
        print("Mining process models...")
        # The numpy operation below is technically deprecated. Ignore warning.
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        splits = np.array_split(log, time_steps)
        models = []
        for partial_log in splits:
            log = EventLog(partial_log)
            model = pm4py.discover_process_tree_inductive(log)
            models.append(model)
        return models

    @staticmethod
    def convert_to_graphs_petri(models) -> List[Data]:
        """
        Converts list of process models into Pytorch Geometric Data graphs
        :param models: list[ProcessTree] of PM4PY process models to be converted to PyG Data graphs
        :return: list[Data] of graphs as PyG Data objects
        """
        # Iterate over process models backwards. Each model is converted to a petri net and
        # - Places and transitions are set as nodes with a numerical one-hot attribute differentiating them
        # - Arcs are converted to edges
        # - The previous model (temporal: next model) is set as "label to train against" y
        print("Converting models to graphs...")
        graphs = []
        previous_graph = None
        for model in reversed(models):
            petri_net = pm4py.convert_to_petri_net(model)
            places = petri_net[0].places
            transitions = petri_net[0].transitions
            arcs = petri_net[0].arcs

            names = []
            node_features = []
            for place in places:
                names.append(place.name)
                node_features.append(0)  # All places in petri net have node feature 0
            for transition in transitions:
                names.append(transition.name)
                node_features.append(1)  # All transitions in petri net have node feature 1
            node_features = torch.tensor(node_features, dtype=torch.long)  # Convert to tensor

            edge_index = []
            for arc in arcs:
                source = names.index(arc.source.name)
                target = names.index(arc.target.name)
                edge_index.append([source, target])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            # print("Added petri net graph, names:" + str(names))      # TODO: Deal with names
            # Shorten transition (not place!) names:
            for i in range(len(names)):
                if node_features[i] == 1:
                    names[i] = "T(" + names[i][:8] + ")"
            graph = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=previous_graph, names=names)
            graphs.append(graph)
            previous_graph = Data(x=graph.x, edge_index=graph.edge_index, names=names)  # Copy without y
        graphs.reverse()  # Reverse list as we want to return it in correct temporal order
        graphs.pop()  # As the last graph does not have anything to train against, it is not part of the training set
        return graphs

    @staticmethod
    def convert_to_graphs_bpmn(models) -> List[Data]:
        """
        Converts list of process models into Pytorch Geometric Data graphs
        :param models: list[ProcessTree] of PM4PY process models to be converted to PyG Data graphs
        :return: list[Data] of graphs as PyG Data objects
        """
        # Iterate over process models backwards. Each model is converted to a bpmn and
        # - Nodes are converted into nodes with a numerical one-hot attribute differentiating them
        # - Arcs are converted to edges
        # - The previous model (temporal: next model) is set as "label to train against" y
        print("Converting models to graphs...")
        graphs = []
        previous_graph = None
        for model in reversed(models):
            bpmn = pm4py.convert_to_bpmn(model)
            graph = bpmn.get_graph()
            nodes = list(graph.nodes)
            node_names = []
            node_features = []
            for node in nodes:
                if type(node) == pm4py.objects.bpmn.obj.BPMN.StartEvent:
                    node_features.append(torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.long))
                elif type(node) == pm4py.objects.bpmn.obj.BPMN.EndEvent:
                    node_features.append(torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.long))
                elif type(node) == pm4py.objects.bpmn.obj.BPMN.Task:
                    node_features.append(torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.long))
                elif type(node) == pm4py.objects.bpmn.obj.BPMN.ExclusiveGateway:
                    node_features.append(torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.long))
                elif type(node) == pm4py.objects.bpmn.obj.BPMN.ParallelGateway:
                    node_features.append(torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.long))
                elif type(node) == pm4py.objects.bpmn.obj.BPMN.InclusiveGateway:
                    node_features.append(torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.long))
                else:
                    print("Found new type: " + str(type(node)))
                    node_features.append(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long))
                node_names.append(node.get_name())
            # print("Node feature vector: " + str(torch.stack(node_features)))

            edges = list(graph.edges)
            edge_index = []
            for edge in edges:
                source = nodes.index(edge[0])
                target = nodes.index(edge[1])
                edge_index.append([source, target])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            graph = Data(x=torch.stack(node_features), edge_index=edge_index.t().contiguous(), names=node_names,
                         y=previous_graph)
            graphs.append(graph)
            previous_graph = Data(x=graph.x, edge_index=graph.edge_index, names=node_names)  # Copy without y
        graphs.reverse()  # Reverse list as we want to return it in correct temporal order
        graphs.pop()  # As the last graph does not have anything to train against, it is not part of the training set
        return graphs

    @staticmethod
    def export_json(graphs):
        """
        Exports provided graphs to json file with syntax appropriate for ProcessModelDriftDataset import
        :param graphs: list[Data] of PyG graphs to be converted to json file
        """
        print("Exporting graphs to json file...")
        data = dict(time_periods=len(graphs), edge_index={}, edge_weights={}, x={}, names={},
                    y={"x": {}, "edge_index": {}, "names": {}})
        max_dim = 0
        for index, graph in enumerate(graphs):
            data["edge_index"][str(index)] = []
            for edge in graph.edge_index:
                data["edge_index"][str(index)].append([int(edge[0]), int(edge[1])])

            # Not currently used, retained for possible future use
            data["edge_weights"][str(index)] = [0] * len(graph.edge_index)

            data["x"][str(index)] = []
            for node in graph.x:
                if type(node) == torch.Tensor:
                    data["x"][str(index)].append(node.tolist())
                else:
                    data["x"][str(index)].append(int(node.item()))
            if len(graph.x) > max_dim:
                max_dim = len(graph.x)

            data["names"][str(index)] = []
            for name in graph.names:
                data["names"][str(index)].append(name)

            data["y"]["edge_index"][str(index)] = []
            data["y"]["x"][str(index)] = []
            data["y"]["names"][str(index)] = []
            for edge_label in graph.y["edge_index"]:
                data["y"]["edge_index"][str(index)].append([int(edge_label[0]), int(edge_label[1])])
            for node_label in graph.y["x"]:
                if node_label.shape == ():
                    data["y"]["x"][str(index)].append(int(node_label.item()))
                else:
                    data["y"]["x"][str(index)].append(node_label.tolist())
            for name in graph.y["names"]:
                data["y"]["names"][str(index)].append(name)
        data["max_dim"] = max_dim + 5
        return json.dumps(data)

    @staticmethod
    def save_json_as_file(json, filename):
        with open(filename, 'w') as outfile:
            outfile.write(json)

    @staticmethod
    def demo_process_tree_one():
        nodes = [
            BPMN.StartEvent(name="Start"),
            BPMN.Task(name="A"),
            BPMN.Task(name="B"),
            BPMN.Task(name="C"),
            BPMN.EndEvent(name="End")
        ]

        flows = [
            BPMN.Flow(source=nodes[0], target=nodes[1]),
            BPMN.Flow(source=nodes[1], target=nodes[2]),
            BPMN.Flow(source=nodes[2], target=nodes[3]),
            BPMN.Flow(source=nodes[3], target=nodes[4])
        ]

        bpmn = BPMN(nodes=nodes, flows=flows)
        process_tree = pm4py.convert_to_process_tree(bpmn)
        return process_tree

    @staticmethod
    def demo_process_tree_two():
        nodes = [
            BPMN.StartEvent(name="Start"),
            BPMN.Task(name="A"),
            BPMN.Task(name="B"),
            BPMN.EndEvent(name="End")
        ]

        flows = [
            BPMN.Flow(source=nodes[0], target=nodes[1]),
            BPMN.Flow(source=nodes[1], target=nodes[2]),
            BPMN.Flow(source=nodes[2], target=nodes[3])
        ]

        bpmn = BPMN(nodes=nodes, flows=flows)
        process_tree = pm4py.convert_to_process_tree(bpmn)
        return process_tree


if __name__ == '__main__':
    # Note: Requires cdlg package! (pip install git+https://gitlab.uni-mannheim.de/processanalytics/cdlg-package)
    print("Starting dataset generation...")

    # Parameters
    num_traces = 200
    start_point = 0.3
    end_point = 0.6
    distribution_type = "linear"  # "linear" or "exponential"
    process_tree_one = None  # ProcessDatasetGenerator.demo_process_tree_one()
    process_tree_two = None  # ProcessDatasetGenerator.demo_process_tree_two()
    change_proportion = 0
    timesteps = 10
    graph_type = "bpmn"  # {"bpmn", "petri_net"}
    repeats = 1
    drift_type = "sudden"  # {"sudden", "gradual", "recurring", "incremental"}
    import_xes = ""

    process_graphs = []
    for i in range(repeats):
        print("Generating process model " + str(i + 1) + " of " + str(repeats) + "...")
        # Generate event log
        if drift_type == "gradual":
            event_log = generate_log_with_gradual_drift(num_traces, start_point, end_point, distribution_type,
                                                        process_tree_one,
                                                        process_tree_two, change_proportion)
        elif drift_type == "sudden":
            event_log = generate_log_with_sudden_drift(num_traces, change_point=end_point, model_one=process_tree_one,
                                                       model_two=process_tree_two, change_proportion=change_proportion)
        elif drift_type == "recurring":
            event_log = generate_log_with_recurring_drift(num_traces=num_traces, start_point=start_point,
                                                          end_point=end_point,
                                                          model_one=process_tree_one, model_two=process_tree_two,
                                                          change_proportion=change_proportion)
        elif drift_type == "incremental":
            if process_tree_one is None:
                process_tree_one = generate_specific_trees('middle')
            event_log = generate_log_with_incremental_drift(traces=[num_traces]*4, change_proportion=change_proportion)
        else:
            raise ValueError("Invalid drift type!")
        xes_exporter.apply(event_log, "./test.xes")
        # Overwrite event log here to import existing instead of generating new
        if import_xes != "":
            event_log = pm4py.read_xes(import_xes)

        # Mining process models on event log
        #   Adding +1 here as the last graph will not be converted into a Data graph object
        #   as that would leave it without any label to train against
        process_models = ProcessDatasetGenerator.mine_process_models(log=event_log, time_steps=timesteps + 1)

        # Converting process models to petri net based graphs
        if graph_type == "petri_net":
            process_graphs += ProcessDatasetGenerator.convert_to_graphs_petri(models=process_models)
        else:
            process_graphs += ProcessDatasetGenerator.convert_to_graphs_bpmn(models=process_models)
    # Exporting graphs as JSON that's compatible with the ProcessModelDriftDataset class
    json_string = ProcessDatasetGenerator.export_json(graphs=process_graphs)
    # Export method: file or printout
    # print(json_string)
    ProcessDatasetGenerator.save_json_as_file(json_string, f"process_dataset_export_{graph_type}.json")
    print("Done!")
