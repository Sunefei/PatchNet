import time

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from .utils import *

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size**-0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        # Different kind of pooling
        # if pooling == "sum":
        #     self.pool = torch.sum
        # elif pooling == "mean":
        #     self.pool = torch.mean
        # elif pooling == "max":
        #     self.pool = torch.max

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


# parser.add_argument('--JK', type=str, default="last", help='how the node features are combined across layers. last, sum, max or concat')


class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act,
                 thresh):
        super(ATT_learner, self).__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.sparse = sparse
        self.mlp_act = mlp_act
        self.thresh = thresh

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features, edge_ori):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            edge_index = torch.stack([rows, cols], dim=0)
            values = apply_non_linearity(values, self.non_linearity, self.i)
            values = torch.nan_to_num(values, nan=0.0)
            values_ = values[torch.nonzero(values).squeeze()]
            values_ = F.normalize(values_, dim=0)
            # print("sometimes wrong", values_)
            len_v = values_.shape[0]
            thresh = self.thresh
            edge_index_ = edge_index[:,
                                     torch.topk(values_, int(thresh *
                                                             len_v))[1]]
            edge_index_resid = torch.unique(torch.cat((edge_index_, edge_ori),
                                                      dim=1),
                                            dim=1)
            edge_attr = values_
            return edge_index_resid, edge_attr
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities,
                                               self.non_linearity, self.i)
            return similarities


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class EncoderLayer(nn.Module):
    def __init__(self, gnn_type, gnn_layer, win_size, step, hidden_size,
                 ffn_size, dropout_rate, attention_dropout_rate, num_heads,
                 pooling, k, sim_function, sparse, activation_learner, thresh):
        super(EncoderLayer, self).__init__()

        self.gnn_layer = gnn_layer
        self.gnn_type = gnn_type
        self.win_size = win_size
        self.step = step

        self.emb_dim = hidden_size

        self.graph_learner = ATT_learner(2, win_size, k, sim_function, 6,
                                         sparse, activation_learner, thresh)

        self.motif_GNN = GNN(gnn_layer,
                             win_size,
                             hidden_size,
                             JK="last",
                             drop_ratio=0.2,
                             gnn_type=gnn_type)
        self.motif_GNN_ori = GNN_ori(gnn_layer,
                                     win_size,
                                     hidden_size,
                                     JK="last",
                                     drop_ratio=0.2,
                                     gnn_type=gnn_type)

        self.attention = Attention(hidden_size)

        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        # Different kind of pooling
        if pooling == "sum":
            self.pool = torch.sum
        elif pooling == "mean":
            self.pool = torch.mean
        elif pooling == "max":
            self.pool = torch.max

    def forward(self, x, edge_index_ori, edge_attr=None, attn_bias=None):
        num_of_token = x.shape[1]
        gnn_out = []
        ori_x = self.motif_GNN_ori(x, edge_index_ori,
                                   edge_attr)
        for layer in range(num_of_token):
            edge_index, edge_attr_ = self.graph_learner(
                x[:, layer, :], edge_index_ori)

            str_x = self.motif_GNN(x[:, layer, :], edge_index,
                                   edge_attr_)
            emb = torch.stack([ori_x[:, layer, :], str_x], dim=1)
            emb, att = self.attention(emb)
            gnn_out.append(emb)

        gnn_out = torch.stack(gnn_out, dim=1)
        y = self.self_norm(gnn_out)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x_mid = gnn_out + y

        y = self.ffn_norm(x_mid)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x_mid + y
        # pooling
        x = self.pool(x, dim=1)
        t4=time.perf_counter()
        return x

    #def from_pretrained(self, model_file):
    #    self.molecule_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    #    return

def Patch(x, win_size, token_size, step):
    assert type(x) == torch.Tensor
    num_node = x.shape[0]
    step = (x.shape[1]-win_size)//token_size
    assert(step<=win_size)
    if step==0:
        step=1
    unfold_res = x.unfold(1, win_size, step)
    return unfold_res, unfold_res.shape




class NewNN_ori(torch.nn.Module):
    def __init__(self, win_size, emb_dim):
        super(NewNN_ori, self).__init__()
        # self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.win_size = win_size
        self.linear = Sequential(Linear(win_size, emb_dim), ReLU(),
                                 Linear(emb_dim, emb_dim))

    def forward(self, x):
        num_of_node = x.shape[0]
        x = x.view(num_of_node, -1, self.win_size)
        num_of_token = x.shape[1]
        x = self.linear(x)
        x = x.view(num_of_node, -1)
        return x


class NewNN(torch.nn.Module):
    def __init__(self, win_size, emb_dim):
        super(NewNN, self).__init__()
        # self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.win_size = win_size
        self.linear = Sequential(Linear(win_size, emb_dim), ReLU(),
                                 Linear(emb_dim, emb_dim))

    def forward(self, x):
        x = self.linear(x)
        return x


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self,
                 num_layer,
                 win_size,
                 emb_dim,
                 JK="last",
                 drop_ratio=0,
                 gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        # self.num_of_token = num_of_token
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.win_size = win_size
        self.emb_dim = emb_dim

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            lay_nn = NewNN(win_size, self.emb_dim)
            self.gnns.append(GINConv(lay_nn))
            win_size = self.emb_dim

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_ori(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self,
                 num_layer,
                 win_size,
                 emb_dim,
                 JK="last",
                 drop_ratio=0,
                 gnn_type="gin"):
        super(GNN_ori, self).__init__()
        self.num_layer = num_layer
        # self.num_of_token = num_of_token
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.win_size = win_size
        self.emb_dim = emb_dim
        # Sequential(Linear(input_dim, emb_dim), ReLU(),
        #                 Linear(emb_dim, emb_dim))
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            lay_nn = NewNN_ori(win_size, self.emb_dim)
            self.gnns.append(GINConv(lay_nn))
            win_size = self.emb_dim

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        num_of_nodes = x.shape[0]
        x = x.reshape(num_of_nodes, -1)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = h.view(num_of_nodes, self.emb_dim, -1)
            h = self.batch_norms[layer](h)

            h = h.permute(0, 2, 1)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:

                h = h.reshape(num_of_nodes, -1)
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 2

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output


class GNN_nodepred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_nodepred, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 2

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        output = self.graph_pred_linear(node_representation)

        return output

if __name__ == "__main__":
    pass

