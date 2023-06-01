import numpy as np
import torch
import torch.optim as optim
from config import args
from dataloader import DataLoaderSubstructContext, DataLoaderMasking
from util import ExtractSubstructureContextPair, cycle_index, MaskAtom

from datasets import MoleculeDataset
from tqdm import tqdm
import time
from multimodel import PretrainModule
import min_norm_solvers
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph

def train(args, device, loader_CP, loader_AM, optimizer):
    curr_losses = {}
    model.train()
    model.zero_grad()
    Loss=0
    step=0
    for batch_CP, batch_AM in zip(tqdm(loader_CP), loader_AM):
        step += 1
        optimizer.zero_grad()
        loss_data = {}
        grads = {}
        batch_CP.to(device)
        batch_AM.to(device)
        batch = {'CP': batch_CP, 'AM': batch_AM}
        tasks = {'CP', 'AM'}
        # -------------- Begin of Pareto Multi-Tasking Learning --------------
        if 'CP' in tasks:
            t = 'CP'
            loss = model.CP(batch[t])
            grads[t] = []
            loss_data[t] = loss.data
            loss.backward()
            for param in model.main_model.parameters():
                if param.grad is not None:
                    grads[t].append(param.grad.data.detach().cpu())
            model.zero_grad()

        if 'AM' in tasks:
            t = 'AM'
            loss = model.AM(batch[t])
            grads[t] = []
            loss_data[t] = loss.data
            loss.backward()
            for param in model.main_model.parameters():
                if param.grad is not None:
                    grads[t].append(param.grad.data.detach().cpu())
            model.zero_grad()

        if len(tasks) > 1:
            gn = min_norm_solvers.gradient_normalizers(grads, loss_data, 'l2')
            for t in loss_data:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
            sol, _ = min_norm_solvers.MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
            sol = {k: sol[i] for i, k in enumerate(tasks)}
        # -------------- End of Pareto Multi-Tasking Learning --------------
        model.zero_grad()
        train_loss = 0
        actual_loss = 0
        loss_dict = model(batch)

        for i, l in loss_dict.items():
            train_loss += float(sol[i]) * l
            actual_loss += l

        train_loss.backward()

        loss_dict['train_loss'] = actual_loss.detach()
        for k, v in sol.items():
            loss_dict[k + '_weight'] = torch.tensor(float(v))
            if k not in curr_losses:
                curr_losses[k] = loss_dict[k].item()
            else:
                curr_losses[k] += loss_dict[k].item()
        if 'train_loss' not in curr_losses:
            curr_losses['train_loss'] = loss_dict['train_loss']
        else:
            curr_losses['train_loss'] += loss_dict['train_loss']

        optimizer.step()
        model.zero_grad()

        Loss += curr_losses['train_loss']
        for k in curr_losses:
            curr_losses[k] = 0


    return Loss / step

if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize
    print('num layer: %d l1: %d l2: %d' % (args.num_layer, l1, l2))

    if args.pretrain_dataset=='zinc':
        dataset_CP = MoleculeDataset(args.dataset_root + args.dataset, dataset=args.dataset,
                                  transform=ExtractSubstructureContextPair(args.num_layer, l1, l2))
        dataset_AM = MoleculeDataset(args.dataset_root + args.dataset, dataset=args.dataset,
                              transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                                 mask_rate=args.mask_rate, mask_edge=args.mask_edge))
    elif args.pretrain_dataset=='pcba':
        dataset_CP = PygGraphPropPredDataset(name='ogbg-molpcba', root='/data/syf/finetune/dataset/',
                                         transform=ExtractSubstructureContextPair(args.num_layer, l1, l2))
        dataset_AM = PygGraphPropPredDataset(name='ogbg-molpcba', root='/data/syf/finetune/dataset/',
                            transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                               mask_rate=args.mask_rate, mask_edge=args.mask_edge))
    elif args.pretrain_dataset=='pcqm':
        dataset_CP = PygPCQM4Mv2Dataset(root='/data/syf/finetune/dataset/', smiles2graph=smiles2graph,
                                    transform=ExtractSubstructureContextPair(args.num_layer, l1, l2))
        dataset_AM = PygPCQM4Mv2Dataset(root='/data/syf/finetune/dataset/', smiles2graph=smiles2graph,
                                    transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                                       mask_rate=args.mask_rate, mask_edge=args.mask_edge))
    if args.pretrain_dataset=='':
        raise 'Pretrain Dataset Invalid'

    loader_CP = DataLoaderSubstructContext(dataset_CP, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
    loader_AM = DataLoaderMasking(dataset_AM, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    model = PretrainModule(args.gnn_type, args.num_layer, args.win_size, args.step,
                        args.emb_dim * 2, args.emb_dim, args.for_dropout,
                        args.dropout, args.gnn_dropout, args.num_heads, args.gat_heads,
                        args.pooling, l1, l2, args.token_size, args.num_tokens, args.pretrain_dataset,
                        args.k, args.sim_function, args.sparse, args.activation_learner, args.thresh).to(device)

    model_param_group = [{'params': model.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train_loss = train(args, device, loader_CP, loader_AM, optimizer)
        print(f'epoch{epoch}: loss={train_loss}')
        if not args.output_model_dir == '':
            torch.save(model.main_model.state_dict(),
                       args.output_model_dir + 'Multie' + str(epoch) + '_model.pth')

    if not args.output_model_dir == '':
        torch.save(model.main_model.state_dict(),
                   args.output_model_dir + 'Multi_model.pth')