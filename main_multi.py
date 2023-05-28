import random

import numpy as np
import torch
import torch.optim as optim
from config import args
from dataloader import DataLoaderSubstructContext, DataLoaderMasking
from util import ExtractSubstructureContextPair, MaskAtom

from datasets import MoleculeDataset
import copy
from tqdm import tqdm
from multimodel import PretrainModule
import min_norm_solvers
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def train(args, device, loader_CP, loader_AM, optimizer):
    curr_losses = {}
    model.train()
    model.zero_grad()
    Loss=0
    step=0
    l = len(loader_CP)
    s = list(range(l))
    random.shuffle(s)
    for id in tqdm(s):
        batch_CP = loader_CP[id]
        batch_AM = loader_AM[id]
        step += 1
        optimizer.zero_grad()
        loss_data = {}
        grads = {}
        batch_CP.to(device)
        batch_AM.to(device)
        batch = {'CP': batch_CP, 'AM': batch_AM}
        tasks = {'CP', 'AM'}
        # -------------- Begin of Pareto Multi-Tasking Learning --------------
        if True:
            if 'CP' in tasks:
                t = 'CP'
                loss = model.module.CP(batch[t])
                grads[t] = []
                loss_data[t] = loss.data
                loss.backward()
                for param in model.module.main_model.parameters():
                    if param.grad is not None:
                        grads[t].append(param.grad.data.detach().cpu())
                model.zero_grad()
            if 'AM' in tasks:
                t = 'AM'
                loss = model.module.AM(batch[t])
                grads[t] = []
                loss_data[t] = loss.data
                loss.backward()
                for param in model.module.main_model.parameters():
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
        else:
            sol = {s:1 for s in tasks}
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

def mix(n, loader_list=[]):
    cnt=0
    mix_dataset = []
    for i in range(n):
        for batch in loader_list[i]:
            mix_dataset.append(copy.deepcopy(batch))
            cnt+=1
    return mix_dataset

if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    dist.init_process_group(backend='nccl')

    device = torch.device('cuda:' + str(args.local_rank)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.local_rank)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize
    print('num layer: %d l1: %d l2: %d' % (args.num_layer, l1, l2))
    datasets_CP = [MoleculeDataset(args.dataset_root + args.dataset, dataset=args.dataset,
                                     transform=ExtractSubstructureContextPair(args.num_layer, l1, l2)),
                   PygGraphPropPredDataset(name='ogbg-molpcba', root=args.dataset_root,
                                           transform=ExtractSubstructureContextPair(args.num_layer, l1, l2)),
                   PygPCQM4Mv2Dataset(root=args.dataset_root, smiles2graph=smiles2graph,
                                      transform=ExtractSubstructureContextPair(args.num_layer, l1, l2))]
    datasets_AM = [MoleculeDataset(args.dataset_root + args.dataset, dataset=args.dataset,
                                     transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                                        mask_rate=args.mask_rate, mask_edge=args.mask_edge)),
                   PygGraphPropPredDataset(name='ogbg-molpcba', root=args.dataset_root,
                                           transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                                              mask_rate=args.mask_rate, mask_edge=args.mask_edge)),
                   PygPCQM4Mv2Dataset(root=args.dataset_root, smiles2graph=smiles2graph,
                                      transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                                                         mask_rate=args.mask_rate, mask_edge=args.mask_edge))]

    sampler = [DistributedSampler(datasets_CP[i]) for i in range(3)]
    loaders_CP = [DataLoaderSubstructContext(datasets_CP[i], batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers, sampler=sampler[i])
                  for i in range(3)]
    loaders_AM = [DataLoaderMasking(datasets_AM[i], batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, sampler=sampler[i])
                 for i in range(3)]

    mixCP = mix(3, loaders_CP)
    mixAM = mix(3, loaders_AM)

    model = PretrainModule(args.gnn_type, args.num_layer, args.win_size, args.step,
                        args.emb_dim * 2, args.emb_dim, args.for_dropout,
                        args.dropout, args.gnn_dropout, args.num_heads, args.gat_heads,
                        args.pooling, l1, l2, args.token_size, args.num_tokens, args.pretrain_dataset,
                        args.k, args.sim_function, args.sparse, args.activation_learner, args.thresh).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    model_param_group = [{'params': model.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train_loss = train(args, device, mixCP, mixAM, optimizer)
        print(f'epoch{epoch}: loss={train_loss}')
        if not args.output_model_dir == '' and args.local_rank == 0:
            torch.save(model.module.main_model.state_dict(),
                       args.output_model_dir + 'Multie' + str(epoch) + '_model.pth')

    if not args.output_model_dir == '':
        torch.save(model.module.main_model.state_dict(),
                   args.output_model_dir + 'Multi_model.pth')