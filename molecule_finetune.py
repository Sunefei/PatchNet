from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models.molecule_gnn_model import EncoderLayer, Patch
from models.molecule_gnn_model import GNN_graphpred
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
from util import get_num_task

from datasets import MoleculeDataset
import copy
from mole.vqvae import VectorQuantizer
from mole.model import GNN
from tqdm import tqdm
import time

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        with torch.no_grad():
            x = copy.deepcopy(batch.x)
            e = copy.deepcopy(batch.edge_attr)
            x = tokenizer(x, batch.edge_index, e)
            x, shape = Patch(x, args.win_size, args.token_size, args.step)
        pred = model(x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            x = copy.deepcopy(batch.x)
            e = copy.deepcopy(batch.edge_attr)
            x = tokenizer(x, batch.edge_index, e)
            x, shape = Patch(x, args.win_size, args.token_size, args.step)
            pred = model(x, batch.edge_index, batch.edge_attr, batch.batch)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print('{} is invalid'.format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


if __name__ == '__main__':
    ans = []
    ans_last = []
    ans_e100 = []
    ans_pos = []
    for runseed in range(10):
        torch.manual_seed(runseed)
        np.random.seed(runseed)
        device = torch.device('cuda:' + str(args.device)) \
            if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runseed)

        # Bunch of classification tasks
        num_tasks = get_num_task(args.dataset)
        dataset_folder = '/data/syf/finetune/dataset/'
        dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
        print(dataset)

        eval_metric = roc_auc_score

        if args.split == 'scaffold':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                      header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(
                dataset, smiles_list, null_value=0, frac_train=args.train_ratio[0],
                frac_valid=args.train_ratio[1], frac_test=args.train_ratio[2])
            print('split via scaffold')
        elif args.split == 'random':
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset, null_value=0, frac_train=args.train_ratio[0],
                frac_valid=args.train_ratio[1], frac_test=args.train_ratio[2], seed=args.seed)
            print('randomly split')
        elif args.split == 'random_scaffold':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                      header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(
                dataset, smiles_list, null_value=0, frac_train=args.train_ratio[0],
                frac_valid=args.train_ratio[1], frac_test=args.train_ratio[2], seed=args.seed)
            print('random scaffold')
        else:
            raise ValueError('Invalid split option.')
        #print(train_dataset[0])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

    # set up model
        molecule_model = EncoderLayer(args.gnn_type, args.num_layer, args.win_size, args.step,
                           args.emb_dim * 2, args.emb_dim, args.for_dropout,
                           args.dropout, args.num_heads,
                           args.pooling, args.k, args.sim_function, args.sparse, args.activation_learner, args.thresh)
        model = GNN_graphpred(args=args, num_tasks=num_tasks,
                              molecule_model=molecule_model)
        if not args.input_model_file == '':
            model.from_pretrained(args.input_model_file)
            print(f'============from {args.input_model_file}=================')
        model.to(device)
        print(model)

        tokenizer = GNN(5, 300,
                        gnn_type=args.gnn_type).to(device)
        codebook = VectorQuantizer(300,
                                   args.num_tokens,
                                   commitment_cost=0.25).to(device)
        dir = './mole/'
        tokenizer.from_pretrained(dir + "checkpoints/vqencoder.pth")
        codebook.from_pretrained(dir + "checkpoints/vqquantizer.pth")

        # set up optimizer
        # different learning rates for different parts of GNN
        if not args.freeze:
            print('-----------------Activated----------------------')
            model_param_group = [{'params': model.molecule_model.parameters()},
                                 {'params': model.graph_pred_linear.parameters(),
                                  'lr': args.lr * args.lr_scale}]
        else:
            print('==================freeezed=================')
            model_param_group = [{'params': model.graph_pred_linear.parameters(),
                                  'lr': args.lr * args.lr_scale}]
        optimizer = optim.Adam(model_param_group, lr=args.lr,
                               weight_decay=args.decay)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        train_roc_list, val_roc_list, test_roc_list = [], [], []
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        best_val_roc, best_val_idx = -1, 0
        best_test_roc, best_test_idx = -1, 0

        for epoch in range(1, args.epochs + 1):
            loss_acc = train(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if args.eval_train:
                train_roc, train_acc, train_target, train_pred = eval(model, device, train_loader)
            else:
                train_roc = train_acc = 0
            val_roc, val_acc, val_target, val_pred = eval(model, device, val_loader)
            test_roc, test_acc, test_target, test_pred = eval(model, device, test_loader)

            train_roc_list.append(train_roc)
            train_acc_list.append(train_acc)
            val_roc_list.append(val_roc)
            val_acc_list.append(val_acc)
            test_roc_list.append(test_roc)
            test_acc_list.append(test_acc)
            print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = epoch - 1
                if not args.output_model_dir == '':
                    output_model_path = join(args.output_model_dir, 'model_best.pth')
                    saved_model_dict = {
                        'molecule_model': molecule_model.state_dict(),
                        'model': model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = join(args.output_model_dir, 'evaluation_best.pth')
                    np.savez(filename, val_target=val_target, val_pred=val_pred,
                             test_target=test_target, test_pred=test_pred)
            if test_roc > best_test_roc:
                best_test_roc = test_roc
                best_test_idx = epoch - 1
        print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx],
                                                                     val_roc_list[best_val_idx],
                                                                     test_roc_list[best_val_idx]))
        print('last train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[-1],
                                                                     val_roc_list[-1],
                                                                     test_roc_list[-1]))
        ans.append(test_roc_list[best_val_idx])
        ans_last.append(test_roc_list[-1])
        ans_e100.append(test_roc_list[5])
        ans_pos.append(best_test_idx)
        if args.output_model_dir is not '':
            output_model_path = join(args.output_model_dir, 'model_final.pth')
            saved_model_dict = {
                'molecule_model': molecule_model.state_dict(),
                'model': model.state_dict()
            }
            torch.save(saved_model_dict, output_model_path)
    m = np.round(sum(ans) / len(ans), decimals=5)
    v = np.round(np.std(ans), decimals=5)
    print('{} Final Result(best): {}({})'.format(args.dataset, m, v))
    m = np.round(sum(ans_last) / len(ans_last), decimals=5)
    v = np.round(np.std(ans_last), decimals=5)
    print('{} Final Result(last): {}({})'.format(args.dataset, m, v))
    m = np.round(sum(ans_e100) / len(ans_e100), decimals=5)
    v = np.round(np.std(ans_e100), decimals=5)
    print('{} Final Result(e5): {}({})'.format(args.dataset, m, v))
    print('{} Final Result(pos): {}'.format(args.dataset, ans_pos))