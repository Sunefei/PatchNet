import numpy as np
import torch
import torch.nn as nn
from config import args
from models import GNN
from models.molecule_gnn_model import EncoderLayer, Patch
from torch_geometric.nn import global_mean_pool
from util import cycle_index
import copy
from mole.vqvae import VectorQuantizer
from mole.model import GNN
from tqdm import tqdm
import time

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)

class PretrainModule(torch.nn.Module):
    def __init__(self, gnn_type, gnn_layer, win_size, step,
                 hidden_size, ffn_size, dropout_rate, attention_dropout_rate, gnn_dropout,
                 gat_heads, num_heads, pooling, l1, l2, token_size, num_tokens, pretrain_dataset,
                 k, sim_function, sparse, activation_learner, thresh):
        super(PretrainModule, self).__init__()
        self.win_size = win_size
        self.token_size = token_size
        self.step = step
        dir = './mole/'
        self.tokenizer = GNN(5, 300, gnn_type='gin')
        self.codebook = VectorQuantizer(300, num_tokens, commitment_cost=0.25)
        self.tokenizer.from_pretrained(dir + "checkpoints/vqencoder.pth")
        self.codebook.from_pretrained(dir + "checkpoints/vqquantizer.pth")
        self.molecule_readout_func = global_mean_pool

        self.main_model = EncoderLayer(gnn_type, gnn_layer, win_size, step,
                                                hidden_size, ffn_size, dropout_rate,
                                                attention_dropout_rate, num_heads,
                                                pooling, k, sim_function, sparse, activation_learner, thresh)
        #========================CP============================================
        self.molecule_context_model = EncoderLayer(gnn_type, int(l2 - l1), win_size, step,
                                                hidden_size, ffn_size, dropout_rate,
                                                attention_dropout_rate, num_heads,
                                                pooling, k, sim_function, sparse, activation_learner, thresh)
        self.criterion_CP = torch.nn.BCEWithLogitsLoss()

        #=======================AM==============================================
        self.molecule_atom_masking_model = torch.nn.Linear(hidden_size, 119)

        self.criterion_AM = torch.nn.CrossEntropyLoss()


    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def compute_representation(self, g, X):
        self.train(False)
        with torch.no_grad():
            h = self.big_model(g, X)
        self.train(True)
        return h.detach()

    def forward(self, batch):
        res = {}
        if 'CP' in batch:
            res['CP'] = self.CP(batch['CP'])

        if 'AM' in batch:
            res['AM'] = self.AM(batch['AM'])

        return res

    def CP(self, batch):
        with torch.no_grad():
            x1 = copy.deepcopy(batch.x_substruct)
            e1 = copy.deepcopy(batch.edge_attr_substruct)
            x1 = self.tokenizer(x1, batch.edge_index_substruct, e1)
            x1, shape = Patch(x1, self.win_size, self.token_size, self.step)
            x2 = copy.deepcopy(batch.x_context)
            e2 = copy.deepcopy(batch.edge_attr_context)
            x2 = self.tokenizer(x2, batch.edge_index_context, e2)
            x2, shape = Patch(x2, self.win_size, self.token_size, self.step)
        # creating substructure representation
        substruct_repr = self.main_model(
            x1, batch.edge_index_substruct,
            batch.edge_attr_substruct)[batch.center_substruct_idx]

        # creating context representations
        overlapped_node_repr = self.molecule_context_model(
            x2, batch.edge_index_context,
            batch.edge_attr_context)[batch.overlap_context_substruct_idx]

        # positive context representation
        # readout -> global_mean_pool by default
        context_repr = self.molecule_readout_func(overlapped_node_repr,
                                             batch.batch_overlapped_context)

        # negative contexts are obtained by shifting
        # the indices of context embeddings
        neg_context_repr = torch.cat(
            [context_repr[cycle_index(len(context_repr), i + 1)]
             for i in range(args.contextpred_neg_samples)], dim=0)

        num_neg = args.contextpred_neg_samples
        pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
        pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

        loss_pos = self.criterion_CP(pred_pos.double(),
                             torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = self.criterion_CP(pred_neg.double(),
                             torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        contextpred_loss = loss_pos + num_neg * loss_neg

        num_pred = len(pred_pos) + len(pred_neg)
        contextpred_acc = (torch.sum(pred_pos > 0).float() +
                           torch.sum(pred_neg < 0).float()) / num_pred
        contextpred_acc = contextpred_acc.detach().cpu().item()

        return contextpred_loss

    def AM(self, batch):
        with torch.no_grad():
            x = copy.deepcopy(batch.masked_x)
            e = copy.deepcopy(batch.edge_attr)
            x = self.tokenizer(x, batch.edge_index, e)
            x, shape = Patch(x, self.win_size, self.token_size, self.step)
        node_repr = self.main_model(x, batch.edge_index, batch.edge_attr)
        target = batch.mask_node_label[:, 0]
        node_pred = self.molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
        attributemask_loss = self.criterion_AM(node_pred.double(), target)
        attributemask_acc = compute_accuracy(node_pred, target)
        return attributemask_loss


    def from_pretrained(self, model_file):
        self.main_model.load_state_dict(torch.load(model_file, map_location='cpu'))

