import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from model.config import cfg
from torch_scatter import scatter_add
from copy import deepcopy
from torch.cuda.amp import autocast

class GCNLayer(nn.Module):

    def __init__(self, dim_in, dim_out, pos_isn):
        super(GCNLayer, self).__init__()
        self.pos_isn = pos_isn
        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
            self.linear_skip_bias = nn.Parameter(torch.ones(size=(dim_out, )))
        elif cfg.gnn.skip_connection == 'identity':
            assert self.dim_in == self.out_channels

        self.linear_msg_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.activate = nn.ReLU()
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)

        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, graph):
        # edge_index = graph.edges()
        edge_index = graph.edges(etype='_E')

        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),),
                                 device=row.device)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt

    def message_fun(self, edges):
        edge_feat = edges.data['edge_feat']
        src_features = edges.src['h']
        dst_features = edges.dst['h']


        mlp = nn.Linear(edge_feat.size(-1), src_features.size(-1)).to(edge_feat.device)
        edge_feat_transformed = mlp(edge_feat)  # [num_edges, src_feat_dim]

        edge_attention = torch.sigmoid(nn.Linear(edge_feat.size(-1), 1).to(edge_feat.device)(edge_feat))
        result = edge_attention * src_features + edge_feat_transformed+ dst_features

        return {'m': result}

    def forward(self, g, feats, edge_feats, fast_weights=None):

        if fast_weights:
            linear_skip_weight = fast_weights[0]
            linear_skip_bias = fast_weights[1]
            linear_msg_weight = fast_weights[2]
            linear_msg_bias = fast_weights[3]

        else:
            linear_skip_weight = self.linear_skip_weight
            linear_skip_bias = self.linear_skip_bias
            linear_msg_weight = self.linear_msg_weight
            linear_msg_bias = self.linear_msg_bias

        feat_src, feat_dst = expand_as_pair(feats, g)

        norm_ = self.norm(g)

        feat_src = feat_src * norm_.view(-1, 1)

        g.srcdata['h'] = feat_src
        num_existing_edges = edge_feats.shape[0]
        total_edges = g.num_edges(etype='_E')

        device = edge_feats.device
        # g = g.to(device)  #
        #
        total_edges_account = g.num_edges(etype='account')
        zero_features_account = torch.zeros((total_edges_account, 4), device=g.device)
        g.edges['account'].data['edge_feat'] = zero_features_account


        # If the number of existing edge features is less than the total number of edges in the graph, expansion is required
        if num_existing_edges < total_edges:
            # Create a zero tensor with the shape of (total_edges - num_existing_edges, 4)
            zero_features = torch.zeros((total_edges - num_existing_edges, 4), device=device)

            # Concatenate the existing edge features and zero tensors
            expanded_edge_feats = torch.cat([edge_feats, zero_features], dim=0)
        else:
            # If the number of existing edge features is already equal to or greater than the total number of edges in the graph, use the existing edge features directly
            expanded_edge_feats = edge_feats

        expanded_edge_feats = expanded_edge_feats.to(g.device)

        g.edges['_E'].data['edge_feat'] = expanded_edge_feats

        g.multi_update_all(
            {
                '_E': (self.message_fun, fn.sum(msg='m', out='h')),

                'account': (self.message_fun, fn.sum(msg='m', out='h'))
            },
            cross_reducer='sum'
        )

        # The features aggregated by the target node after message passing have subjected the features of the target node to a norm similar to that of the source node
        rst = g.dstdata['h']
        rst = rst * norm_.view(-1, 1)

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)

        #Skip joins allow information to be passed directly from the previous layer to the subsequent layer
        if cfg.gnn.skip_connection == 'affine':
            skip_x = F.linear(feats, linear_skip_weight, linear_skip_bias)
        elif cfg.gnn.skip_connection == 'identity':
            skip_x = feats
        return rst_ + skip_x

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, num_layers, dropout):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        # GCN
        for i in range(num_layers):
            d_in = in_features if i == 0 else out_features
            pos_isn = True if i == 0 else False
            layer = GCNLayer(d_in, out_features, pos_isn)
            self.add_module('layer{}'.format(i), layer)

        self.weight1 = nn.Parameter(torch.ones(size=(hidden_dim, out_features)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, hidden_dim)))

        # Edge decoding calculation method
        if cfg.model.edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif cfg.model.edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)
        else:
            raise ValueError('Unknown edge decoding {}.'.format(
                cfg.model.edge_decoding))
        # Model initialization parameters
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, e, label_edge_index, fast_weights=None):
        count = 0
        for layer in self.children():
            if fast_weights is None:
                x = layer(g, x, e)
            else:
                x = layer(g, x, e, fast_weights[2 + count * 4: 2 + (count + 1) * 4])
            count += 1


        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        # The node feature normalization is stored in g.node_embedding.
        x = F.normalize(x)
        g.node_embedding = x

        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = F.sigmoid(F.linear(pred, weight2))

        label_edge_index = torch.tensor(label_edge_index)

        node_feat = pred[label_edge_index]

        nodes_first = node_feat[0]
        nodes_second = node_feat[1]

        pred1 = self.decode_module(nodes_first, nodes_second)

        return pred1






