import glob
import os

import pandas
import pm4py
import torch
import torch_geometric

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.algorithm import Variants

from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.exporter import exporter as petri_net_exporter
from pm4py.visualization.process_tree import visualizer as pt_visualizer
import xml.etree.ElementTree as ET

from torch_geometric.data import Data

"""
Collection of PM4PY process mining services
- .csv to .xes
- .xes to PM4PY Eventlog
- Eventlog to ProcessTree
- ProcessTree to .bpmn
- ProcessTree to .pnml
- Petri Net (Places, Transitions, Arcs, Names?) to (Graph, Names?)
- Graph to dot
- log to batch files
    - Input: log, traces_per_batch, batches_receptive_field, first_batch_size
    - Output: .xes and .pnml files for every batch
- batch files to list of Data objects (PyG graphs)
    - Input: .pnml files for every batch
    - Output: List of Data objects (PyG graphs)
"""

class Place:
    def __init__(self, identifier, name):
        self.id = identifier
        self.name = name

    def __str__(self):
        return "Place[" + str(self.id) + " " + self.name + "]"


class Transition:
    def __init__(self, identifier, name):
        self.id = identifier
        self.name = name

    def __str__(self):
        return "Transition[" + str(self.id) + " " + self.name + "]"


class Arc:
    def __init__(self, identifier, source, target):
        self.id = identifier
        self.source = source
        self.target = target

    def __str__(self):
        return "Arc[" + self.id + " " + str(self.source) + " " + str(self.target) + "]"


def import_csv(file_path):
    # event_log = pm4py.format_dataframe(pandas.read_csv('./process-datasets/running-example.csv', sep=';'), case_id='case_id',
    #                                    activity_key='activity', timestamp_key='timestamp')
    event_log = pandas.read_csv(file_path, sep=';')
    pm4py.write_xes(event_log, '../../process-datasets/running-example-exported.xes')
    num_events = len(event_log)
    num_cases = len(event_log.case_id.unique())
    print("Number of events: {}\nNumber of cases: {}".format(num_events, num_cases))


def import_xes(file_path):
    return pm4py.read_xes(file_path)


def alpha_miner(event_log):
    # Variants: Variants.IM - default, Variants.IMf - better model without replay guarantee, Variants.IMd - fastest, worse model
    tree = inductive_miner.apply_tree(event_log, variant=Variants.IMf)
    gviz = pt_visualizer.apply(tree)
    pt_visualizer.view(gviz)
    return tree


def export_bpmn(tree):
    bpmn = bpmn_exporter.apply(tree, "../../process-datasets/export.bpmn")
    return bpmn


def export_pnml(tree, path="./process-datasets/petri_net.pnml"):
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    petri_net = petri_net_exporter.apply(net, output_filename=path, initial_marking=initial_marking, final_marking=final_marking)
    # pt_visualizer.view(petri_net)
    return petri_net


def pnml_to_petri_net(path="./process-datasets/petri_net.pnml"):
    places = []
    transitions = []
    arcs = []
    names = {}
    tree = ET.parse(path)
    root = tree.getroot()
    petri_net = root[0][0]

    # Create places, transitions and arcs based on the pnml
    # ID needs to be a number, which is why the original ID is stored as the name attribute instead
    for child in petri_net:
        if child.tag == 'place':
            places.append(Place((len(places) + len(transitions)), child.attrib.get("id")))
            names[child.attrib.get("id")] = (len(places) + len(transitions))
        elif child.tag == 'transition':
            transitions.append(Transition((len(places) + len(transitions)), child.attrib.get("id")))
            names[child.attrib.get("id")] = (len(places) + len(transitions))
        elif child.tag == 'arc':
            arcs.append(Arc(child.attrib.get("id"), names[child.attrib.get("source")], names[child.attrib.get("target")]))
    # print(" ".join(str(e) for e in places))
    # print(" ".join(str(e) for e in transitions))
    # print(" ".join(str(e) for e in arcs))
    return places, transitions, arcs, names


# Limitations:
# - edge_features should be arc IDs but tensors can only be numbers (and they aren't important anyway)
def petri_net_to_graph(places, transitions, arcs, names=None, previous_graph=None):
    node_features = [0] * (len(places) + len(transitions))
    for place in places:
        node_features[place.id] = 0
    for transition in transitions:
        node_features[transition.id] = 1
    node_features = torch.tensor(node_features, dtype=torch.long)
    # print(node_features)

    edge_index = []
    for arc in arcs:
        edge_index.append([arc.source, arc.target])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # print(edge_index)
    return Data(x=node_features, edge_index=edge_index.t().contiguous(), y=previous_graph), names


def graph_to_dot(graph, names=None, graph_name="G"):
    res = dict((v, k) for k, v in names.items())
    print("digraph "+ graph_name +" {")
    i = 1
    for node in graph.x:
        if node == 0:
            try:
                print("    " + str(i) + "[shape=ellipse label=\"" + str(res[i]) + "\"]")
            except KeyError:
                print("    " + str(i) + "[shape=ellipse label=\"" + str(i) + "\"]")
        else:
            try:
                print("    " + str(i) + "[shape=box label=\"" + str(res[i]) + "\"]")
            except KeyError:
                print("    " + str(i) + "[shape=box label=\"" + str(i) + "\"]")
        i += 1
    for a in graph.edge_index:
        print("    " + str(a[0].item()) + "->" + str(a[1].item()))
    print("}")


# Convert an event log (.xes) to a dataset consisting of
# traces_per_batch traces of log split into batches, with each batch being matched to a process model mined on the last batches_receptive_field batches
# TODO: Handle overhanging traces (that are not fitted into final batch)
# TODO: This currently assumes that the event log is ordered by timestamp!
# Arguments:
# - event_log: the event log to convert
# - traces_per_batch: the number of traces per batch
# - batches_receptive_field: the number of batches to "remember" (i.e. the number of batches to use for mining the process model)
# - first_batch_size: the number of traces in the first batch (should usually be larger than traces_per_batch)
def log_to_batch_files(log, traces_per_batch, batches_receptive_field, first_batch_size=0, path="./process_dataset"):
    if first_batch_size == 0:
        first_batch_size = traces_per_batch
    trace_count = 0
    batch_count = 0
    current_batch = EventLog()
    for trace in log:
        # If batch is not full, add trace to batch
        if trace_count < traces_per_batch or (trace_count < first_batch_size and batch_count == 0):
            current_batch.append(trace)
            trace_count += 1
        # If batch is full, add batch to dataset, mine model and start new batch
        else:
            trace_count = 0
            pm4py.write_xes(current_batch, path+'/batch_' + str(batch_count) + '.xes')
            current_batch = EventLog()
            # Mine model on last batches_to_remember batches
            log = EventLog()
            for i in range(batches_receptive_field):
                if batch_count >= i:
                    temp_log = pm4py.read_xes(path+'/batch_' + str(batch_count - i) + '.xes')
                    for temp_trace in temp_log:
                        log.append(temp_trace)
            tree = alpha_miner(log)
            export_pnml(tree, path=path+"/model_" + str(batch_count) + ".pnml")
            batch_count += 1


def batch_files_to_datalist(path):
    dataset = []  # List of Data objects containing graphs of the petri nets of the mined models
    previous_graph = None  # Graph of the last mined model ()

    # list_of_files = sorted(filter(os.path.isfile,
    #                               glob.glob(path + '*')))

    files = glob.glob(path+"/*")
    files.sort(key=os.path.getmtime)
    print("Iterating over files: "+str(files))
    for file in files:
        filename = os.fsdecode(file)
        if filename.endswith(".pnml"):  # Only read pnml files
            places, transitions, arcs, names = pnml_to_petri_net(filename)
            if previous_graph is not None:
                dataset.append(petri_net_to_graph(places, transitions, arcs, names, previous_graph)[0])
            previous_graph = petri_net_to_graph(places, transitions, arcs, names)[0]
    return dataset


"""
TODOs:
- Graph generation
  - Look into general/theoretical methods
  - See if PyG has anything

Very basic idea:
- Process model as graph
- Input is process log at time x_t
    - Problem: How to represent the process log?
    - Training dataset: Log at time x_t + Process model at time x_t
    - Loss: Entropy measures on graphs
- Output is process model at time x_t+1 <- This should remain constant 99% of the time
"""
if __name__ == "__main__":
    # ---- LOGS ----
    # log = pm4py.read_xes('./process-datasets/running-example-exported.xes')  # Import log
    # log = pm4py.read_xes("./process-datasets/pdc_2016/pdc_2016_1.xes")
    log = pm4py.read_xes("../../process-datasets/BPI_Challenge_2017.xes")
    # ---- LOGS ----

    # -- Flow from log to dot export of process model petri net ----
    # tree = alpha_miner(log)  # Mine process model
    # export_pnml(tree)  # Export process model to pnml
    # places, transitions, arcs, names = pnml_to_petri_net()  # Convert pnml to petri net
    # graph, names = petri_net_to_graph(places, transitions, arcs, names=names)  # Convert petri net to graph
    # graph_to_dot(graph, names=names)  # Export graph to dot
    # -- Flow from log to dot export of process model petri net ----

    # -- log to dataset ----
    # This will take the log specified above and
    # - split it into batches of 600 traces (exception: first batch is 1000 traces)
    # - mine a process model on the last 10 batches for each batch
    log_to_batch_files(log, traces_per_batch=600, batches_receptive_field=10, first_batch_size=1000)
    # -- log to dataset ----
