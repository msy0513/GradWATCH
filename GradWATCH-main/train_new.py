import torch
import random
import numpy as np
from copy import deepcopy
from model.loss import prediction, Link_loss_meta
from model.utils import report_rank_based_eval_meta
from model.utils import compute_mrr

def train(args, model, optimizer, device, graph_l, labels, labels_edge_index, logger, n):
    best_param = {'best_auc': 0, 'best_state': None, 'best_s_dw': None}
    earl_stop_c = 0
    epoch_count = 0

    for epoch in range(args.epochs):
        all_auc = 0.0
        i = 0
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))

        S_dw = [0] * len(fast_weights)

        train_count = 0

        while i < (n - args.window_num):
            if i != 0:
                i = random.randint(i, i + args.window_num)
            if i >= (n - args.window_num):
                break
            graph_train = graph_l[i: i + args.window_num]
            labels_train = labels[i: i + args.window_num]

            labels_edge_index_train = labels_edge_index[i: i + args.window_num]

            i = i + 1
            # Copy a version of data as a valid in the window
            features = [graph_unit.node_feature.to(device) for graph_unit in graph_train]
            features_edge = [graph_unit.edge_feature.to(device) for graph_unit in graph_train]

            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            window_auc = 0.0
            losses = torch.tensor(0.0).to(device)
            count = 0
            # one window
            for idx, (graph, label, label_edge_index) in enumerate(zip(graph_train, labels_train, labels_edge_index_train)):
                # The last snapshot in the window is valid only
                if idx == args.window_num - 1:
                    break

                # t snapshot train
                # Copy a version of data as a train in the window
                feature_node_train = deepcopy(features[idx])
                feature_edge_train = deepcopy(features_edge[idx])

                graph = graph.to(device)

                label = label.to(device)

                label_edge_index = label_edge_index.to(device)

                pred = model(graph, feature_node_train, feature_edge_train, label_edge_index, fast_weights)
                # pred, candidate_set = model(graph, feature_node_train, feature_edge_train, label_edge_index, fast_weights)

                loss = Link_loss_meta(pred, label)

                # t grad
                grad = torch.autograd.grad(loss, fast_weights)

                beta = args.beta

                S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))

                fast_weights = list(
                    map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

                # t+1 snapshot valid
                graph_train[idx + 1] = graph_train[idx + 1].to(device)
                labels_train[idx + 1] = labels_train[idx + 1].to(device)
                labels_edge_index_train[idx + 1] = labels_edge_index_train[idx + 1].to(device)

                # pred, candidate_set = model(graph_train[idx + 1], features[idx + 1], features_edge[idx + 1], labels_edge_index_train[idx + 1], fast_weights)
                pred = model(graph_train[idx + 1], features[idx + 1], features_edge[idx + 1],
                                            labels_edge_index_train[idx + 1], fast_weights)

                loss = Link_loss_meta(pred, labels_train[idx + 1])

                label = labels_train[idx + 1]
                label_edge_index = labels_edge_index_train[idx + 1]
                # pos:neg = 1:20
                mrr, _, _, _ = report_rank_based_eval_meta(model, graph_train[idx + 1], features[idx+1], features_edge[idx + 1], label, label_edge_index, fast_weights)

                # rec, prec, f1, auc, hr1, hits = prediction(pred, labels_train[idx + 1],candidate_set, labels_edge_index_train[idx + 1])
                rec, prec, f1, auc, fpr, fnr = prediction(pred, labels_train[idx + 1])

                droprate = torch.FloatTensor(np.ones(shape=(1)) * args.drop_rate)

                masks = torch.bernoulli(1. - droprate).unsqueeze(1)

                if masks[0][0]:
                    losses = losses + loss
                    count += 1
                    window_auc += auc
                logger.info('meta epoch:{}, mrr:{:.5f}, loss: {:.5f}, rec: {:.5f}, prec: {:.5f}, f1: {:.5f}, auc: {:.5f}'.
                            format(epoch, mrr, loss, rec, prec, f1, auc))

            if losses:
                losses = losses / count
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            if count:
                all_auc += window_auc / count
            train_count += 1

        all_auc = all_auc / train_count
        epoch_count += 1

        if all_auc > best_param['best_auc']:
            best_param = {'best_auc': all_auc, 'best_state': deepcopy(model.state_dict()), 'best_s_dw': S_dw}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    return best_param