import matplotlib.pyplot as plt
import networkx as nx
import pm4py
import numpy as np
import torch
import torch_geometric
from pm4py.objects.log.obj import EventLog
# from legacy import process_mining_services as pm
from torch.nn import BCELoss
import os.path as osp



def loss(value):
    return abs((value - 0.5)%1)

if __name__ == '__main__':
    # print(str(loss(1)))
    # print(str(loss(0)))
    # print(str(loss(0.5)))

    # path = osp.join("..", "BPI_Challenge_2012.xes")
    # log = pm4py.read_xes(path)
    # model = pm4py.discover_process_tree_inductive(log)
    # bpmn = pm4py.convert_to_bpmn(model)
    # pm4py.view_bpmn(bpmn)

    edge_index = torch.tensor([[0,1],
                               [1,2]], dtype=torch.long)
    x = torch.tensor([[1],[2],[3]], dtype=torch.float)

    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data)
    nx.draw_networkx(g, font_color="white")
    plt.show()


    # # log = pm4py.read_xes("process-datasets/BPI_Challenge_2012.xes")
    # # # log = pm4py.read_xes("process-datasets/log.xes")
    # # splits = np.array_split(log, 50)  # Vary batch size here
    # # base_tree = pm4py.discover_process_tree_inductive(EventLog(splits[0]))
    # #
    # # for index, partial_log in enumerate(splits):
    # #     process_tree = pm4py.discover_process_tree_inductive(EventLog(partial_log))
    # #     print("For "+str(index)+":")
    # #
    # #     print("Left dif:")
    # #     print(list(set(base_tree.children) - set(process_tree.children)))
    # #     print(len(list(set(base_tree.children) - set(process_tree.children))))
    # #     print("Right dif:")
    # #     print(list(set(process_tree.children) - set(base_tree.children)))
    # #     print(len(list(set(process_tree.children) - set(base_tree.children))))
    # #
    # #     print("Children:")
    # #     print(dir(process_tree))
    # #     print(process_tree.children)
    # #     print("Label")
    # #     print(process_tree.label)
    # #     # print("Dir:")
    # #     # print(dir(process_tree))
    # #     # bpmn_model = pm4py.convert_to_bpmn(process_tree)
    # #     # filename = "BPI-2012-2-"+str(index)+".png"
    # #     # pm4py.save_vis_bpmn(bpmn_model, "./bpmn-visualizations/50/"+filename)
    # #     # print("Exported "+filename)
    # #
    # #     pm4py.convert_to_petri_net(process_tree)
    # # log = pm4py.read_xes("process-datasets/generated_logs/terminal_log_1660249688.xes")
    # log = pm4py.read_xes("../process-datasets/generated_logs/terminal_log_1660249688.xes")
    # # log = pm4py.read_xes("process-datasets/generated_logs/terminal_log_1660251479.xes")
    # # log = pm4py.read_xes("process-datasets/generated_logs/terminal_log_1660252258.xes")
    # # log = pm4py.read_xes("process-datasets/generated_logs/terminal_log_1660252430.xes")
    # print(len(log))
    # splits = np.array_split(log, 3)  # Vary batch size here
    # base_tree = pm4py.discover_process_tree_inductive(EventLog(splits[0]))
    #
    # for index, partial_log in enumerate(splits):
    #     process_tree = pm4py.discover_process_tree_inductive(EventLog(partial_log))
    #     print("For "+str(index)+":")
    #     print("Left dif:")
    #     print(list(set(base_tree.children) - set(process_tree.children)))
    #     print(len(list(set(base_tree.children) - set(process_tree.children))))
    #     print("Right dif:")
    #     print(list(set(process_tree.children) - set(base_tree.children)))
    #     print(len(list(set(process_tree.children) - set(base_tree.children))))
    #     bpmn_model = pm4py.convert_to_bpmn(process_tree)
    #     filename = "cdlg-"+str(index)+".png"
    #     pm4py.save_vis_bpmn(bpmn_model, "./bpmn-visualizations/"+filename)
    #     print("Exported "+filename)
