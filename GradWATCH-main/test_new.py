from random import random

import torch
import numpy as np
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta
from model.utils import compute_mrr


def test(graph_l, labels, labels_edge_index, model, args, logger, n, S_dw, device):
    beta = args.beta
    # avg_auc = avg_rec = avg_prec = avg_f1 = avg_mrr = avg_fpr = avg_fnr = 0.0

    graph_test = graph_l[n:]
    labels_test = labels[n:]
    labels_edge_index_test = labels_edge_index[n:]

    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    num_graphs = len(graph_test) - 1  # 计算的时间步数量

    auc_list, rec_list, prec_list, f1_list, mrr_list, fpr_list, fnr_list, all_hr_5, all_avg_hr_a = [], [], [], [], [], [], [], [],[]

    for idx, (g_test, label_test, label_edge_index_test) in enumerate(
            zip(graph_test, labels_test, labels_edge_index_test)):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature)
        edege_feature_train = deepcopy(g_test.edge_feature)
        graph_train = graph_train.to(device)

        g_test = g_test.to(device)
        label_test = label_test.to(device)
        label_edge_index_test = label_edge_index_test.to(device)

        # pred, candidate_set = model(g_test, graph_train, edege_feature_train, label_edge_index_test, fast_weights)
        pred = model(g_test, graph_train, edege_feature_train, label_edge_index_test, fast_weights)
        loss = Link_loss_meta(pred, label_test)

        grad = torch.autograd.grad(loss, fast_weights)

        S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0].pow(2), zip(grad, S_dw)))

        fast_weights = list(
            map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

        graph_test[idx + 1] = graph_test[idx + 1].to(device)
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature.to(device)
        labels_test[idx + 1] = labels_test[idx + 1].to(device)
        labels_edge_index_test[idx + 1] = labels_edge_index_test[idx + 1].to(device)


        pred = model(graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                    graph_test[idx + 1].edge_feature,
                                    labels_edge_index_test[idx + 1], fast_weights)


        label = labels_test[idx + 1]
        label_edge_index = labels_edge_index_test[idx + 1]
        mrr, _, _, _ = report_rank_based_eval_meta(model, graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                                   graph_test[idx + 1].edge_feature, label, label_edge_index,
                                                   fast_weights)
        labels_test[idx + 1] = label
        labels_edge_index_test[idx + 1] = label_edge_index

        rec, prec, f1, auc, ht5, hits = prediction(pred, labels_test[idx + 1])

        auc_list.append(auc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)
        mrr_list.append(mrr)
        all_avg_hr_a.append(hits)
        all_hr_5.append(ht5)

        print("mrr:", mrr, "rec:", rec, "prec:", prec, "f1:", f1, "auc:", auc, "hit5:", ht5, "hits:", hits)

        print("Before logger: graph_test length:", len(graph_test))

        logger.info(
            'meta test, mrr: {:.5f}, rec: {:.5f}, prec: {:.5f}, f1: {:.5f}, auc: {:.5f}, hit1: {:.5f}, hits: {:.5f},'.
            format(mrr, rec, prec, f1, auc, ht5, hits))

    print("Debug: graph_test length:", len(graph_test))

    logger.info("Debug: graph_test length: {}".format(len(graph_test)))

    avg_auc = sum(auc_list) / num_graphs
    avg_rec = sum(rec_list) / num_graphs
    avg_prec = sum(prec_list) / num_graphs
    avg_f1 = sum(f1_list) / num_graphs
    avg_mrr = sum(mrr_list) / num_graphs
    avg_hr_5 = sum(all_hr_5) / num_graphs
    all_avg_hr_a = sum(all_avg_hr_a) / num_graphs

    logger.info({'avg_auc': avg_auc, 'avg_rec': avg_rec, 'avg_prec': avg_prec, 'avg_f1': avg_f1, 'avg_mrr': avg_mrr, 'avg_fpr': avg_hr_5,
                 'avg_fnr': all_avg_hr_a})

    return avg_auc, avg_rec, avg_prec, avg_f1, avg_mrr, avg_hr_5, all_avg_hr_a

