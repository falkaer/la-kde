import argparse
import os
import os.path as osp

import numpy as np
import torch

from experiments.run_utils import run_all
from lakde import *
from datasets import *

import ax
from ax.storage.json_store.load import load_experiment
from ax.service.utils.best_point import get_best_parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        choices=['gas', 'power', 'hepmass', 'bsds300', 'miniboone', 'pinwheel', '2spirals',
                                 'checkerboard'])
    parser.add_argument('model', type=str, choices=['hierarchical', 'knn', 'full', 'diag', 'scalar'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bsize', type=int, default=10_000)
    parser.add_argument('--sparsity_threshold', type=float, default=5e-4)
    return parser.parse_args()

SAVE_ROOT = 'hparams'
all_num_train = [100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000]
num_subsets = 5

def get_closest_hparams(model, dataset, num_train):
    # find largest available hparam run with N <= num_train
    ind = next(i for i, x in reversed(list(enumerate(all_num_train))) if x <= num_train)
    while ind >= 0:
        num_train = all_num_train[ind]
        path = osp.join(SAVE_ROOT, model, dataset, '{}_{}_{}_experiment.json'.format(model, dataset, num_train))
        if osp.exists(path):
            experiment = load_experiment(path)
            return get_best_parameters(experiment, ax.Models)[0]
        else:
            ind -= 1 # try next if it isn't found

if __name__ == '__main__':
    args = parse_args()
    
    print('Running curves for {} on {} at {} sparsity levels'.format(args.model, args.dataset, args.sparsity_threshold))
    
    def model_supplier(num_train):
        print('Instantiating a {} model...'.format(args.model))
        if args.model in ['knn', 'hierarchical']:
            hparams = get_closest_hparams(args.model, args.dataset, num_train)
        
        if args.model == 'knn':
            print('Found hparams: nu_0={}, k={}'.format(hparams['nu_0'], hparams['k']))
            model = LocalKNearestKDE(nu_0=hparams['nu_0'], k_or_knn_graph=hparams['k'],
                                     block_size=args.bsize, verbose=True, logs=False)
        elif args.model == 'hierarchical':
            print('Found hparams: nu_0={}'.format(hparams['nu_0']))
            model = LocalHierarchicalKDE(nu_0=hparams['nu_0'], k_or_knn_graph=min(num_train - 1, 500),
                                         block_size=args.bsize, verbose=True, logs=False)
        elif args.model == 'full':
            model = SharedFullKDE(block_size=args.bsize, verbose=True, logs=False)
        elif args.model == 'diag':
            model = SharedDiagonalizedKDE(block_size=args.bsize, verbose=True, logs=False)
        elif args.model == 'scalar':
            model = SharedScalarKDE(block_size=args.bsize, verbose=True, logs=False)
        return model
    
    if args.dataset in ['pinwheel', '2spirals', 'checkerboard']:
        rng = np.random.RandomState(args.seed)
        train_data = inf_train_gen(args.dataset, rng, batch_size=1_000_000)
        test_data = inf_train_gen(args.dataset, rng, batch_size=50_000)
        dataset_name = args.dataset
        ll_rtol = 1e-4
    else:
        if args.dataset == 'gas':
            dataset = GAS()
            ll_rtol = 1e-4
        elif args.dataset == 'power':
            dataset = POWER()
            ll_rtol = 1e-4
        elif args.dataset == 'hepmass':
            dataset = HEPMASS()
            ll_rtol = 4e-3
        elif args.dataset == 'bsds300':
            dataset = BSDS300()
            ll_rtol = 1e-2
        elif args.dataset == 'miniboone':
            dataset = MINIBOONE()
            ll_rtol = 2e-3
        
        train_data = dataset.trn.x
        test_data = dataset.val.x
        dataset_name = dataset.__class__.__name__
    
    metrics = run_all(train_data, test_data, model_supplier, all_num_train, num_subsets,
                      threshold=args.sparsity_threshold, ll_rtol=ll_rtol)
    
    path = 'curves ({})/{}/{}_curves.pt'.format(args.sparsity_threshold, args.model, args.dataset)
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(metrics, path)
