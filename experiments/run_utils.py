from collections import defaultdict

import numpy as np
import torch

import signal

from lakde.callbacks import ELBOCallback
from lakde.kdes import AbstractKDE
from lakde.local_adaptive_kde import LocallyAdaptiveKDE

def run_kde_model(model, train_data, test_data, threshold, ll_rtol=1e-4, iterations=1000, validate_every=1):
    train_data = torch.from_numpy(train_data).float().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()
    
    N = train_data.shape[0]
    best_ll = -np.inf
    max_active_rnms = 0
    
    def best_log_likelihood(X, model, iter_step):
        nonlocal best_ll
        num_blocks = (N - 1) // model.block_size + 1
        epoch = iter_step // num_blocks
        batch = iter_step % num_blocks
        
        if batch + 1 != num_blocks or (epoch + 1) % validate_every != 0:
            return
        
        avg_ll = model.log_pred_density(X, test_data).mean().item()
        print(', Test log likelihood: {:<10f}'.format(avg_ll), end='')
        
        if avg_ll > best_ll:
            if best_ll != -np.inf and avg_ll - best_ll < ll_rtol:
                print('\nReached relative (likelihood) tolerance cutoff at {} iterations'.format(iter_step))
                best_ll = max(avg_ll, best_ll)
                raise KeyboardInterrupt
            else:
                best_ll = avg_ll
        else:
            raise KeyboardInterrupt
    
    def maximum_active_responsibilities(X, model, iter_step):
        nonlocal max_active_rnms
        if hasattr(model, 'rnm_active_contribs'):
            active_rnms = model.rnm_active_contribs.sum().float() / N ** 2
        else:
            active_rnms = 0
        if active_rnms > max_active_rnms:
            max_active_rnms = active_rnms
    
    callbacks = []
    if isinstance(model, LocallyAdaptiveKDE):
        callbacks.append(ELBOCallback())
    callbacks.extend([maximum_active_responsibilities, best_log_likelihood])
    
    model.fit(train_data, iterations, callbacks, threshold)
    return best_ll, {'max_active': max_active_rnms}

# model is corrected
def run_k_radius_model(model, train_data, test_data):
    from experiments.plot_utils import kde_mesh, trapz_grid
    train_data = torch.from_numpy(train_data).float().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()
    
    corrected = model.corrected
    
    try:
        
        # numerically find the normalization constant, only works in 2 dimensions
        model.corrected = True
        corrected_const = trapz_grid(*kde_mesh(model, train_data))
        corrected_ll = model.log_pred_density(train_data, test_data).mean().cpu().item()
        
        model.corrected = False
        uncorrected_const = trapz_grid(*kde_mesh(model, train_data))
        uncorrected_ll = model.log_pred_density(train_data, test_data).mean().cpu().item()
    
    finally:
        model.corrected = corrected
    
    return corrected_ll - np.log(corrected_const), {'corrected_ll'     : corrected_ll,
                                                    'corrected_const'  : corrected_const,
                                                    'uncorrected_ll'   : uncorrected_ll,
                                                    'uncorrected_const': uncorrected_const}

def get_subset(dataset, size, seed):
    rng = np.random.default_rng(seed)
    sample_size = min(len(dataset), size)
    inds = rng.choice(len(dataset), sample_size, replace=False)
    return dataset[inds]

def run_model(model, *args, **kwargs):
    if isinstance(model, AbstractKDE):
        return run_kde_model(model, *args, **kwargs)
    else:
        return run_k_radius_model(model, *args, **kwargs)

def run_all(train_data, test_data, model_supplier, sample_sizes, num_subsets, *args, **kwargs):
    metrics = defaultdict(list)
    
    def noop_handler(sig, frame):
        pass
    
    old_handler = signal.signal(signal.SIGINT, noop_handler)
    
    try:
        for sample_size in sample_sizes:
            for k in range(num_subsets):
                if train_data.shape[0] < sample_size:
                    sample_size = train_data.shape[0]
                
                # still sample a subset just for the shuffle
                X = get_subset(train_data, sample_size, sample_size + k)
                model = model_supplier(sample_size)
                
                print('Training model on {} sample subset ({}/{})'.format(sample_size, k + 1, num_subsets))
                log_density, meta = run_model(model, X, test_data, *args, **kwargs)
                print('Finished with log density {}'.format(log_density))
                
                metrics[str(sample_size)].append(log_density)
                
                if meta is not None:
                    for k, v in meta.items():
                        metrics[str(sample_size) + '_' + k].append(v)
        
        return metrics
    finally:
        signal.signal(signal.SIGINT, old_handler)
