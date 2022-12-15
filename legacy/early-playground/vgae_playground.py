import os.path as osp

import mlflow as mlflow

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
# from dataset_featurizer import MoleculeDataset
# from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser()
# parser.add_argument('--variational', action='store_true')
# parser.add_argument('--linear', action='store_true')
# parser.add_argument('--dataset', type=str, default='Cora',
#                     choices=['Cora', 'CiteSeer', 'PubMed'])
# parser.add_argument('--epochs', type=int, default=400)
# args = parser.parse_args()


# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels)
#         self.conv2 = GCNConv(2 * out_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)
#
#
# class VariationalGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels)
#         self.conv_mu = GCNConv(2 * out_channels, out_channels)
#         self.conv_logstd = GCNConv(2 * out_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
#
# class LinearEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         return self.conv(x, edge_index)
from legacy.petri.petri_dataset import ProcessModelPetriDataset


# class VariationalLinearEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_mu = GCNConv(in_channels, out_channels)
#         self.conv_logstd = GCNConv(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
#
# # in_channels, out_channels = dataset.num_features, 16
# # if not args.variational and not args.linear:
# #     model = GAE(GCNEncoder(in_channels, out_channels))
# # elif not args.variational and args.linear:
# #     model = GAE(LinearEncoder(in_channels, out_channels))
# # elif args.variational and not args.linear:
# #     model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
# # elif args.variational and args.linear:
# #     model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
#
# def train():
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(train_data.x, train_data.edge_index)
#     loss = model.recon_loss(z, train_data.pos_edge_label_index)
#     loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#     return float(loss)
#
#
# @torch.no_grad()
# def test(process-datasets):
#     model.eval()
#     z = model.encode(process-datasets.x, process-datasets.edge_index)
#     return model.test(z, process-datasets.pos_edge_label_index, process-datasets.neg_edge_label_index)
#
#
# if __name__ == '__main__':
#     epochs = 1000
#
#     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#
#     transform = T.Compose([
#         T.NormalizeFeatures(),
#         T.ToDevice(device),
#         T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                           split_labels=True, add_negative_train_samples=False),
#     ])
#     path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'process-datasets', 'Planetoid')
#     dataset = Planetoid(path, 'Cora', transform=transform)
#     train_data, val_data, test_data = dataset[0]
#
#     # dataset = ProcessModelDataset(root=path)
#     # train_data = dataset.get_train_data()
#     # val_data = dataset.get_val_data()
#     # test_data = dataset.get_test_data()
#
#     # in_channels, out_channels = dataset.num_features, 16
#     in_channels, out_channels = dataset.num_features, dataset.num_features
#
#     model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
#
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     print(type(dataset[0]))
#
#     for epoch in range(1, epochs + 1):
#         loss = train()
#         auc, ap = test(test_data)
#         print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')


# TODO Repo: https://github.com/deepfindr/gnn-project

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the process-datasets
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss / step


def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'process-datasets/images/cm_{epoch}.png')
    mlflow.log_artifact(f"process-datasets/images/cm_{epoch}.png")


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")



def run_one_training(params):
    params = params[0]
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Loading the dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="process-datasets/", filename="HIV_train_oversampled.csv")
        test_dataset = MoleculeDataset(root="process-datasets/", filename="HIV_test.csv", test=True)
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # < 1 increases precision, > 1 recall
        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        # Start training
        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(300):
            if early_stopping_counter <= 10:  # = x * 5
                # Training
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]


if __name__ == '__main__':
    # mlflow.set_tracking_uri("http://localhost:5000")
    dataset = ProcessModelPetriDataset(root=osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'process-datasets', 'Planetoid'))
    graphs = dataset.get_train_data()
    i = 0
    for graph in graphs:
        print("x", graph.x.shape)
        print("y", graph.y.shape)
        print("edge_index", graph.edge_index.shape)
