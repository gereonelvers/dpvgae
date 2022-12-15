import statistics

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score, roc_curve, auc, \
    roc_auc_score
import torch
from tabulate import tabulate
import os.path as osp
import torch_geometric.transforms as T
from torch import sigmoid

from bpmn_dataset import ProcessModelBPMNDataset
from layers import DirectedProcessVGAE, DirectedEdgeEncoder
from previous_iterations.process_gcn import ProcessGCN


# Yes, I know this is a bit of a mess. I'm sorry.
# This class is used to evaluate ProcessGCN and DPVGAE models.
# It requires correspondingly named model exports to be present.
def evaluate(model):
    # evaluate the model on the evaluation dataset
    y_true = []
    y_pred = []
    adj_true = []
    adj_pred = []
    with torch.no_grad():
        first = True
        for data in val_data:
            y = torch.argmax(data.y, dim=1).to(torch.int32).cpu().numpy()
            y_true.append(y)
            adj_hat, x_hat = None, None
            if isinstance(model, ProcessGCN):
                adj_hat, x_hat = model(data.x.to(device), data.edge_index.to(device))  # Get predictions
            elif isinstance(model, DirectedProcessVGAE):
                adj_hat, x_hat = model(data.to(device))
                adj_hat = sigmoid(adj_hat)
            if first:
                first = False
                print("x", x_hat)
                print("adj", adj_hat)
            max_a, ids = torch.max(x_hat, 1, keepdim=True)  # Get the most likely node type
            x_out = torch.zeros_like(x_hat)  # Create a one-hot encoding
            x_out.scatter_(1, ids, max_a)  # Set the most likely node type to 1
            x_out[x_out != 0] = 1  # Set all other node types to 0
            x_out = torch.argmax(x_out, dim=1).to(torch.int32).cpu().numpy()
            y_pred.append(x_out)  # Add the prediction to the list

            adj_true.append(data.adj_matrix_label.to(torch.int32).cpu().numpy())
            adj_hat = torch.where(adj_hat > 0.1, 1, 0)
            adj_hat = adj_hat.to(torch.int32).cpu().numpy()
            adj_pred.append(adj_hat)

    node_f1_scores = []
    node_precision_scores = []
    node_recall_scores = []
    node_accuracy_scores = []
    for i in range(0, len(y_true)):
        # Calculate the F1 score
        precision = precision_score(y_true[i], y_pred[i], average="macro", zero_division=0)
        node_precision_scores.append(precision)
        recall = recall_score(y_true[i], y_pred[i], average="macro", zero_division=0)
        node_recall_scores.append(recall)
        accuracy = accuracy_score(y_true[i], y_pred[i])
        node_accuracy_scores.append(accuracy)
        # F1 is calculated by hand because the sklearn implementation produced nonsensical results
        f1 = 2 * ((precision * recall) / (precision + recall))
        node_f1_scores.append(f1)

    edge_f1_scores = []
    edge_precision_scores = []
    edge_recall_scores = []
    edge_accuracy_scores = []
    for i in range(0, len(y_true)):
        # Calculate the F1 score
        precision = precision_score(adj_true[i].flatten(), adj_pred[i].flatten(), average="macro", zero_division=1)
        edge_precision_scores.append(precision)
        recall = recall_score(adj_true[i].flatten(), adj_pred[i].flatten(), average="macro", zero_division=1)
        edge_recall_scores.append(recall)
        accuracy = accuracy_score(adj_true[i].flatten(), adj_pred[i].flatten())
        edge_accuracy_scores.append(accuracy)
        # F1 is calculated by hand because the sklearn implementation produced nonsensical results
        f1 = 2 * ((precision * recall) / (precision + recall))
        edge_f1_scores.append(f1)

    return sum(node_f1_scores) / len(node_f1_scores), sum(node_precision_scores) / len(node_precision_scores), sum(
        node_recall_scores) / len(node_recall_scores), sum(node_accuracy_scores) / len(node_accuracy_scores), sum(
        edge_f1_scores) / len(edge_f1_scores), sum(edge_precision_scores) / len(edge_precision_scores), sum(
        edge_recall_scores) / len(edge_recall_scores), sum(edge_accuracy_scores) / len(
        edge_accuracy_scores)


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000, sci_mode=False)

    dataset_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data-dump', 'dpvgae')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # TODO: uncomment once working
    dataset = ProcessModelBPMNDataset(dataset_path, transform=T.ToDevice(device))
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()
    val_data = dataset.get_val_data()

    baseline_model = ProcessGCN(node_channels=6, edge_channels=dataset.max_dim).to(device)
    eval_model = DirectedProcessVGAE(
        edge_encoder=DirectedEdgeEncoder(in_channels=6, out_channels=dataset.max_dim,
                                         hidden_channels=32),
    ).to(device)

    # TODO: Automatically loading different model versions after one another would be nice
    baseline_model.load_state_dict(torch.load('processgcn.pth', map_location=device))
    eval_model.load_state_dict(torch.load('dpvgae.pth', map_location=device))

    baseline_model.eval()
    eval_model.eval()
    gcn_node_f1 = []
    gcn_node_precision = []
    gcn_node_recall = []
    gcn_node_accuracy = []
    gcn_edge_f1 = []
    gcn_edge_precision = []
    gcn_edge_recall = []
    gcn_edge_accuracy = []
    gcn_edge_auc_roc = []

    for i in range(1):
        print("i", i)
        node_f1, node_precision, node_recall, node_accuracy, edge_f1, edge_precision, edge_recall, edge_accuracy, auc_roc = evaluate(
            baseline_model)
        gcn_node_f1.append(node_f1)
        gcn_node_precision.append(node_precision)
        gcn_node_recall.append(node_recall)
        gcn_node_accuracy.append(node_accuracy)
        gcn_edge_f1.append(edge_f1)
        gcn_edge_precision.append(edge_precision)
        gcn_edge_recall.append(edge_recall)
        gcn_edge_accuracy.append(edge_accuracy)
        gcn_edge_auc_roc.append(auc_roc)

    format_string = "{:.4f}"

    gcn_node_f1 = format_string.format(statistics.mean(gcn_node_f1))
    gcn_node_precision = format_string.format(statistics.mean(gcn_node_precision))
    gcn_node_recall = format_string.format(statistics.mean(gcn_node_recall))
    gcn_node_accuracy = format_string.format(statistics.mean(gcn_node_accuracy))
    gcn_edge_f1 = format_string.format(statistics.mean(gcn_edge_f1))
    gcn_edge_precision = format_string.format(statistics.mean(gcn_edge_precision))
    gcn_edge_recall = format_string.format(statistics.mean(gcn_edge_recall))
    gcn_edge_accuracy = format_string.format(statistics.mean(gcn_edge_accuracy))

    dpvgae_node_f1 = []
    dpvgae_node_precision = []
    dpvgae_node_recall = []
    dpvgae_node_accuracy = []
    dpvgae_edge_f1 = []
    dpvgae_edge_precision = []
    dpvgae_edge_recall = []
    dpvgae_edge_accuracy = []
    dpvgae_edge_auc_roc = []

    for i in range(1):
        print("i", i)
        node_f1, node_precision, node_recall, node_accuracy, edge_f1, edge_precision, edge_recall, edge_accuracy, auc_roc = evaluate(
            eval_model)
        dpvgae_node_f1.append(node_f1)
        dpvgae_node_precision.append(node_precision)
        dpvgae_node_recall.append(node_recall)
        dpvgae_node_accuracy.append(node_accuracy)
        dpvgae_edge_f1.append(edge_f1)
        dpvgae_edge_precision.append(edge_precision)
        dpvgae_edge_recall.append(edge_recall)
        dpvgae_edge_accuracy.append(edge_accuracy)
        dpvgae_edge_auc_roc.append(auc_roc)

    dpvgae_node_f1 = format_string.format(statistics.mean(dpvgae_node_f1))
    dpvgae_node_precision = format_string.format(statistics.mean(dpvgae_node_precision))
    dpvgae_node_recall = format_string.format(statistics.mean(dpvgae_node_recall))
    dpvgae_node_accuracy = format_string.format(statistics.mean(dpvgae_node_accuracy))
    dpvgae_edge_f1 = format_string.format(statistics.mean(dpvgae_edge_f1))
    dpvgae_edge_precision = format_string.format(statistics.mean(dpvgae_edge_precision))
    dpvgae_edge_recall = format_string.format(statistics.mean(dpvgae_edge_recall))
    dpvgae_edge_accuracy = format_string.format(statistics.mean(dpvgae_edge_accuracy))
    dpvgae_edge_auc_roc = format_string.format(statistics.mean(dpvgae_edge_auc_roc))

    header = ['Model', 'Node F1', 'Node Precision', 'Node Recall', 'Node Accuracy', 'Edge F1', 'Edge Precision',
              'Edge Recall', 'Edge Accuracy']
    rows = [
        ['Model', 'Node F1', 'Node Precision', 'Node Recall', 'Node Accuracy', 'Edge F1', 'Edge Precision',
         'Edge Recall', 'Edge Accuracy'],
        ['ProcessGCN', gcn_node_f1, gcn_node_precision, gcn_node_recall, gcn_node_accuracy, gcn_edge_f1,
         gcn_edge_precision, gcn_edge_recall, gcn_edge_accuracy],
        ['DPVGA', dpvgae_node_f1, dpvgae_node_precision, dpvgae_node_recall, dpvgae_node_accuracy, dpvgae_edge_f1,
         dpvgae_edge_precision, dpvgae_edge_recall, dpvgae_edge_accuracy]
    ]

    transposed_rows = zip(*rows)
    print(type(gcn_edge_f1))
    print(tabulate(transposed_rows, tablefmt='latex', floatfmt=".3f"))
