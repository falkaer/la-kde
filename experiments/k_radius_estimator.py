import argparse

import math
import torch
import numpy as np

from datasets import *

# radius r, dimension d
def log_hypersphere_volume(r, d):
    log_unit_vol = d / 2 * math.log(math.pi) - torch.lgamma(torch.tensor(d / 2 + 1))
    return log_unit_vol + d * torch.log(r)

class KNNDensityEstimator:
    def __init__(self, k, p=2, bsize=2000, corrected=True, const=None):
        self.k = k
        self.p = p
        self.bsize = bsize
        self.corrected = corrected
        self.const = const
    
    def log_pred_density(self, X, Y):
        m, d = Y.shape
        n = X.size(0)
        log_probs = torch.empty(m, device=X.device)
        coeff_k = self.k - 1 if self.corrected else self.k
        if coeff_k < 1:  # undefined
            return torch.fill_(log_probs, np.nan)
        const = -math.log(self.const) if self.const else 0
        coeff = const + math.log(coeff_k) - math.log(n)
        zero = torch.zeros((), device=X.device)
        for i in range(0, m, self.bsize):
            Yt = Y[i:i + self.bsize]
            dists = torch.cdist(Yt, X, p=self.p)
            topk = dists.topk(self.k, dim=1, largest=False, sorted=True)[0]
            r = topk[:, -1]
            
            # if the distance is 0 set the probability to 1
            log_probs[i:i + self.bsize] = torch.where(r != 0, coeff - log_hypersphere_volume(r, d), zero)
        
        return log_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        choices=['gas', 'power', 'hepmass', 'bsds300', 'miniboone', 'pinwheel', '2spirals',
                                 'checkerboard'])
    parser.add_argument('k', type=int)
    parser.add_argument('--num_train', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bsize', type=int, default=500)
    args = parser.parse_args()
    
    args.dataset = 'bsds300'
    args.num_train = 500000000000000
    args.seed = 0
    args.k = 2
    
    print('Running with dataset {} and k = {}'.format(args.dataset, args.k))
    
    if args.dataset in ['pinwheel', '2spirals', 'checkerboard']:
        rng = np.random.RandomState(args.seed)
        train_data = inf_train_gen(args.dataset, rng, batch_size=args.num_train)
        test_data = inf_train_gen(args.dataset, rng, batch_size=10_000)
    else:
        if args.dataset == 'gas':
            dataset = GAS()
        elif args.dataset == 'power':
            dataset = POWER()
        elif args.dataset == 'hepmass':
            dataset = HEPMASS()
        elif args.dataset == 'bsds300':
            dataset = BSDS300()
        elif args.dataset == 'miniboone':
            dataset = MINIBOONE()
        
        train_data = dataset.trn.x
        test_data = dataset.val.x
    
    if train_data.shape[0] > args.num_train:
        np.random.seed(args.seed)
        inds = np.random.choice(len(train_data), args.num_train, replace=False)
        train_data = train_data[inds]
    
    # pinwheel
    # 100:   k=3
    # 250:   k=4
    # 500:   k=5
    # 1000:  k=6
    # 2500:  k=8
    # 5000:  k=10
    # 10000: k=15
    # 25000: k=20
    # 50000: k=25
    
    # 2spirals
    # 100:   k=2
    # 250:   k=3
    # 500:   k=4
    # 1000:  k=5
    # 2500:  k=7
    # 5000:  k=8
    # 10000: k=10
    # 25000: k=15
    # 50000: k=19
    
    # checkerboard
    # 100:   k=3
    # 250:   k=4
    # 500:   k=5
    # 1000:  k=6
    # 2500:  k=7
    # 5000:  k=8
    # 10000: k=10
    # 25000: k=15
    # 50000: k=20
    
    print(train_data.shape)
    print(test_data.shape)
    
    X = torch.from_numpy(train_data).float().cuda()
    Y = torch.from_numpy(test_data).float().cuda()
    
    model = KNNDensityEstimator(args.k, bsize=args.bsize, corrected=True)
    log_pdf = model.log_pred_density(X, Y).mean().item()
    print('Corrected log pdf estimate:', log_pdf)
    # const = trapz_grid(*grid_density(model, X)).item()
    # print('Corrected density integrates to:', const)
    # print('Numerically corrected log pdf:', log_pdf - math.log(const))
    
    # model = KNNDensityEstimator(args.k, bsize=args.bsize, corrected=False)
    # log_pdf = model.log_pred_density(X, Y).mean().item()
    # print('Uncorrected log pdf estimate:', log_pdf)
    # 
    # const = trapz_grid(*grid_density(model, X)).item()
    # print('Uncorrected density integrates to:', const)
    # print('Numerically corrected log pdf:', log_pdf - math.log(const))
    
    # model = KNNDensityEstimator(args.k, bsize=args.bsize, corrected=model.corrected, const=const)
    # from toy_contours import plot_kde_colormesh
    # import matplotlib.pyplot as plt
    
    # norm = plt.Normalize(vmin=0, vmax=0.3, clip=True)
    # plot_kde_colormesh(model, X, norm=norm)
    
    # from vbkde import HierarchicalKDE
    # model = HierarchicalKDE(nu_0=4.001, k_or_knn_graph=40, block_size=4000, verbose=True)
    # model.fit(X, iterations=50)
    # plot_kde_colormesh(model, X)
    # 
    # # compare to LA-KDE
    # 
    # from vbkde import LocalFullKDE
    # model = LocalFullKDE(nu_0=50, k_or_knn_graph=100, block_size=2048, verbose=True)
    # model.fit(X, iterations=20, Y=Y, validate_every=5)
    # print('LA-KDE log pdf:', model.log_pred_density(X, Y).mean().item())
    # plot_kde_colormesh(model, X)
    