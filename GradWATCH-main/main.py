from datetime import datetime
import dgl
import math
import wandb
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from model import GradWATCH
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_label
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ens-20', help='dataset used: tornado-rule or ens')

    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')

    parser.add_argument('--seed', type=int, default=2024, help='split seed')

    parser.add_argument('--repeat', type=int, default=1, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=64, help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type')

    parser.add_argument('--lr', type=float, default=0.02, help='initial learning rate.')

    parser.add_argument('--maml_lr', type=float, default=0.008, help='meta learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 loss on parameters).')

    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop meta loss 0.16')

    parser.add_argument('--num_layers', type=int, default=2, help='GNN layer num')

    parser.add_argument('--num_hidden', type=int, default=256, help='number of hidden units of MLP')

    parser.add_argument('--window_num', type=int, default=1, help='windows size')

    parser.add_argument('--dropout', type=float, default=0.1, help='GNN dropout')

    parser.add_argument('--residual', type=bool, default=False, help='skip connection')

    parser.add_argument('--beta', type=float, default=0.89,
                        help='The weight of adaptive learning rate component accumulation')

    parser.add_argument('--node_embedding', type=float, default=8,
                        help='Each feature dimension when training the model end-to-end')


    args = parser.parse_args()
    logger = getLogger(cfg.log_path)

    # load mixing transaction dataset
    graphs, e_feat, e_time, n_feat = load(args.dataset)
    n_dim = n_feat[0].shape[1]
    logger.info(f"n_dim: {n_dim}")
    # n_dim = 16
    n_node = n_feat[0].shape[0]

    # load heterogeneious edge label and transaction edge index
    labels, labels_edge_index = load_label(args.dataset)

    for index, (label, label_edge_index) in enumerate(zip(labels, labels_edge_index)):
        labels[index] = torch.Tensor(label).long()
        labels_edge_index[index] = torch.Tensor(label_edge_index).long()


    def split_labels_by_time(labels, labels_edge_index, train_ratio=0.6, seed=42):
        """
        Divide the labeled data set by time slices to ensure that the test set contains at least one positive sample and one negative sample.
        Args:
            labels (list): The label list or tensor of each time slice
            labels_edge_index (list): The edge index list of each time slice.
            train_ratio (float): Proportion of training data.
            seed (int): Random seed.

        Returns:
            train_labels (list), train_edge_indices (list), test_labels (list), test_edge_indices (list)
        """
        random.seed(seed)
        torch.manual_seed(seed)

        train_labels = []
        train_edge_indices = []
        test_labels = []
        test_edge_indices = []

        for time_idx, (time_labels, time_edge_index) in enumerate(zip(labels, labels_edge_index)):
            if not isinstance(time_labels, torch.Tensor):
                time_labels = torch.tensor(time_labels)
            if not isinstance(time_edge_index, torch.Tensor):
                time_edge_index = torch.tensor(time_edge_index)

            num_samples = len(time_labels)
            if num_samples == 0:
                print(f"Time Slice {time_idx}: No sample，Jump！")
                continue

            pos_indices = torch.where(time_labels == 1)[0]
            neg_indices = torch.where(time_labels == 0)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                print(f"Time Slice {time_idx}: There are only single-category samples. Skip!")
                continue

            pos_train_size = max(1, int(train_ratio * len(pos_indices)))
            neg_train_size = max(1, int(train_ratio * len(neg_indices)))

            pos_indices = pos_indices[torch.randperm(len(pos_indices))]
            neg_indices = neg_indices[torch.randperm(len(neg_indices))]

            train_pos_indices = pos_indices[:pos_train_size]
            test_pos_indices = pos_indices[pos_train_size:]

            train_neg_indices = neg_indices[:neg_train_size]
            test_neg_indices = neg_indices[neg_train_size:]

            if len(test_pos_indices) == 0:
                test_pos_indices = train_pos_indices[-1:].clone()
                train_pos_indices = train_pos_indices[:-1]

            if len(test_neg_indices) == 0:
                test_neg_indices = train_neg_indices[-1:].clone()
                train_neg_indices = train_neg_indices[:-1]

            train_indices = torch.cat([train_pos_indices, train_neg_indices])
            test_indices = torch.cat([test_pos_indices, test_neg_indices])


            train_labels.append(time_labels[train_indices])
            train_edge_indices.append(time_edge_index[:, train_indices])

            test_labels.append(time_labels[test_indices])
            test_edge_indices.append(time_edge_index[:, test_indices])

            print(f"Time Slice {time_idx}: "
                  f"Compose the associated edge samples={len(train_indices)} (a={len(train_pos_indices)}, n={len(train_neg_indices)}), "
                  f"Learn to calculate the loss samples={len(test_indices)} (a={len(test_pos_indices)}, n={len(test_neg_indices)})")

        return train_labels, train_edge_indices, test_labels, test_edge_indices


    train_labels, train_labels_edge_index, test_labels, test_labels_edge_index = split_labels_by_time(
        labels, labels_edge_index, train_ratio=0.6
    )


    filtered_labels_edge_index = []

    for label, label_edge_index in zip(train_labels, train_labels_edge_index):
        label = torch.Tensor(label).long()
        label_edge_index = torch.Tensor(label_edge_index).long()

        mask = (label == 1)

        filtered_edge_index = (
            label_edge_index[0][mask],
            label_edge_index[1][mask]
        )

        filtered_labels_edge_index.append(filtered_edge_index)


    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_avg_auc, all_avg_rec, all_avg_prec, all_avg_f1, all_avg_mrr ,all_avg_fpr, all_avg_fnr= [], [], [], [], [], [], []
    # all_avg_auc, all_avg_rec, all_avg_prec, all_avg_f1, all_avg_mrr, all_avg_hr_a, all_hr_5 = [], [], [], [], [], [],[]

    best_auc = 0.0
    best_model = None
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for rep in range(args.repeat):

        logger.info('num_layers:{}, num_hidden: {}, lr: {}, maml_lr:{}, window_num:{}, drop_rate:{}, 负样本采样固定'.
                    format(args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.window_num, args.drop_rate))


        graph_l = []
        # Data set processing
        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)

            num_nodes = graph_d.num_nodes()  # 获取节点总数

            src, dst = graph_d.edges()
            src_, dst_ = filtered_labels_edge_index[idx]

            src_account_all = torch.cat([src_, dst_])   #source node
            dst_account_all = torch.cat([dst_, src_])   #target node

            # Create a heterogeneous graph containing the original '_E' edge and the predefined 'account' edge
            graph_d = dgl.heterograph({
                ('node', '_E', 'node'): (src, dst),
                ('node', 'account', 'node'): (src_account_all, dst_account_all),  #Set the account side to a two-way side
                # ('node', 'account', 'node'): ([],[]),
                # ('node', 'self_loop', 'node'): ([], [])
            }, num_nodes_dict={'node': num_nodes})
            # -------------------

            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])

            # Check whether the number of nodes or features in the figure matches
            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])

            graph_d = dgl.remove_self_loop(graph_d, etype='_E')
            graph_d = dgl.add_self_loop(graph_d, etype='_E')

            graph_l.append(graph_d)


        # model initialization
        model = GradWATCH.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout)
        model.train()

        # LightDyG optimizer
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        model = model.to(device)

        # It is divided into multiple Windows,
        # each of which is meta updated in addition to the meta training window
        # Partition dataset
        n = math.ceil(len(graph_l) * 0.6)

        # train
        best_param = train(args, model, optimizer, device, graph_l, test_labels, test_labels_edge_index, logger, n)



        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        # test
        model.eval()

        avg_auc, avg_rec, avg_prec, avg_f1, avg_mrr, avg_fpr, avg_fnr = test(graph_l, test_labels,
                                                                       test_labels_edge_index, model, args,
                                                                       logger, n, S_dw, device)

        if avg_auc > best_auc:
            best_model = best_param['best_state']

        all_avg_auc.append(avg_auc)
        all_avg_rec.append(avg_rec)
        all_avg_prec.append(avg_prec)
        all_avg_f1.append(avg_f1)
        all_avg_mrr.append(avg_mrr)
        # all_avg_hr_a.append(hits)
        # all_hr_5.append(ht_5)
        all_avg_fpr.append(avg_fpr)
        all_avg_fnr.append(avg_fnr)


    torch.save(best_model, './model/params/model.pkl')

    all_avg_auc = np.array(all_avg_auc)
    all_avg_rec = np.array(all_avg_rec)
    all_avg_prec = np.array(all_avg_prec)
    all_avg_f1 = np.array(all_avg_f1)
    all_avg_mrr = np.array(all_avg_mrr)
    all_avg_fpr = np.array(all_avg_fpr)
    all_avg_fnr = np.array(all_avg_fnr)


    print(f"all_avg_auc: {np.mean(all_avg_auc)} ± {np.std(all_avg_auc)}\n"
              f"all_avg_rec: {np.mean(all_avg_rec)} ± {np.std(all_avg_rec)}\n"
              f"all_avg_prec: {np.mean(all_avg_prec)} ± {np.std(all_avg_prec)}\n"
              f"all_avg_f1: {np.mean(all_avg_f1)} ± {np.std(all_avg_f1)}\n"
              f"all_avg_mrr: {np.mean(all_avg_mrr)} ± {np.std(all_avg_mrr)}\n"
             f"all_avg_fpr: {np.mean(all_avg_fpr)} ± {np.std(all_avg_fpr)}\n"
              f"all_avg_fnr: {np.mean(all_avg_fnr)} ± {np.std(all_avg_fnr)}"
          )
