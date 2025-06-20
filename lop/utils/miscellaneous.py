import math
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch
from scipy.linalg import svd
import matplotlib.pyplot as plt
from lop.utils.plot_online_performance import generate_online_performance_plot
import sys
import json
import pickle
import argparse
import os
import numpy as np


def net_init(net, orth=0, w_fac=1.0, b_fac=0.0):
    if orth:
        for module in net:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if hasattr(module, 'bias'):
                nn.init.constant_(module.bias, val=0)
    else:
        net[-1].weight.data.mul_(w_fac)
        if hasattr(net[-1], 'bias'):
            net[-1].bias.data.mul_(b_fac)

def fc_body(act_type, o_dim, h_dim, bias=True):
    activation = {'Tanh': nn.Tanh, 'ReLU': nn.ReLU, 'elu': nn.ELU, 'sigmoid':nn.Sigmoid}[act_type]
    module_list = nn.ModuleList()
    if len(h_dim) == 0:
        return module_list
    module_list.append(nn.Linear(o_dim, h_dim[0], bias=bias))
    module_list.append(activation())
    for i in range(len(h_dim) - 1):
        module_list.append(nn.Linear(h_dim[i], h_dim[i + 1], bias=bias))
        module_list.append(activation())
    return module_list


def get_configurations(params: {}):
    # get all parameter configurations for individual runs
    list_params = [key for key in params.keys() if type(params[key]) is list]
    param_values = [params[key] for key in list_params]
    hyper_param_settings = list(itertools.product(*param_values))
    return list_params, hyper_param_settings

def bin_m_errs_np_arr(errs, m=10000):
    mses = []
    for j in tqdm(range(int(errs.shape[0]/m))):
        mses.append(errs[j*m:(j+1)*m].mean())
    return np.array(mses)


def bin_m_errs(errs, m=10000):
    mses = []
    for j in tqdm(range(int(errs.shape[0]/m))):
        mses.append(errs[j*m:(j+1)*m].mean())
    return torch.tensor(mses)


def gaussian_init(net, std_dev=1e-1):
    for module in net:
        if hasattr(module, 'weight'):
            nn.init.normal_(module.weight, mean=0.0, std=std_dev)
        if hasattr(module, 'bias'):
            nn.init.normal_(module.bias, mean=0.0, std=std_dev)


def kaiming_init(net, act='relu', bias=True):
    if act == 'elu':
        act = 'relu'
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            nn.init.kaiming_uniform_(module.weight, nonlinearity=act.lower())
            if bias:
                module.bias.data.fill_(0.0)
    nn.init.kaiming_uniform_(net[-1].weight, nonlinearity='linear')
    if bias:
        net[-1].bias.data.fill_(0.0)


def xavier_init(net, act='tanh', bias=True):
    if act == 'elu':
        act = 'relu'
    gain = nn.init.calculate_gain(act.lower(), param=None)
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if bias:
                module.bias.data.fill_(0.0)
    nn.init.xavier_uniform_(net[-1].weight, gain=1)
    if bias:
        net[-1].bias.data.fill_(0.0)


def lecun_init(net, bias=True):
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            new_bound = math.sqrt(3/module.in_features)
            nn.init.uniform_(module.weight, a=-new_bound, b=new_bound)
            if bias:
                module.bias.data.fill_(0.0)
    new_bound = math.sqrt(3/net[-1].in_features)
    nn.init.uniform_(net[-1].weight, a=-new_bound, b=new_bound)
    if bias:
        net[-1].bias.data.fill_(0.0)


def register_hook(net, hook_fn):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            register_hook(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_forward_hook(hook_fn)


def nll_accuracy(out, yb):
    predictions = torch.argmax(out, dim=1)
    return (predictions == yb).float().mean()


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in tqdm(range(0, inputs.shape[0], batchsize)):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)    # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)

def get_label(algo):
    return {'bp': 'BP', 'cbp': 'CBP', 'l2': 'L2', 'snp': 'SnP', 'cbp_l2': 'CBP+L2', 'cbp_snp': 'CBP+SnP'}[algo]

def get_color(algo):
    return {'bp': 'C0', 'cbp': 'C1', 'l2': 'C2', 'snp': 'C3', 'cbp_l2': 'C6', 'cbp_snp': 'C9'}[algo]

def normalize_array(arr):
    arr = np.array(arr)
    """Normalize a single array to the range [0, 1]"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Check if the array has a non-zero range to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(arr)  # Return zeros if array is constant
    
    # Scale to [0, 1]
    return (arr - min_val) / (max_val - min_val)

def create_histogram(util_data, dir_path='plots', file_prefix='non-normalized_util_data', title='title', normalize=False, divide_average_by=1, 
                     x_max=5.0, y_max=5.0):
    # Prepare data
    data = np.array(util_data) # Assuming util_data is a list of numpy arrays
    # print(f'{title}, data: {data[:20]}')
    # data = np.array([t.numpy() for t in util_data])
    if normalize:
        data = np.array([normalize_array(arr) for arr in data])
    data = data.flatten()
    # print(f'{title}, data: {data[:20]}')

    # plt.close('all') # in case some other plot is open

    fig, ax = plt.subplots(figsize=(10, 6))

    if normalize:
        x_max = 1.0
    bin_width = x_max / 60.
    # ax.set_xlim(0, 2.4)
    #assert(np.max(data) <= x_max)
    ax.set_ylim(0, y_max)

    if (np.max(data) > x_max):
        print(f'Warning: max value {np.max(data)} is greater than x_max {x_max}.')

    # num_bins = 40
    bins = np.arange(0, x_max + bin_width, bin_width)
    weights = np.ones_like(data) / divide_average_by if divide_average_by > 1 else None
    hist, bins, patches = ax.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, weights=weights)
    
    if (hist.max() > y_max):
        print(f'Warning: max histogram value {hist.max()} is greater than y_max {y_max}.')

    ax.grid(axis='y', alpha=0.75, linestyle='--')
    # ax.axvline(np.mean(util_data), color='red', linestyle='dashed', linewidth=2, label='Mean')
    
    if divide_average_by > 1:
        title = f'{title} (Averaged over {divide_average_by} runs)'
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    # plt.show()

    filepath = os.path.join(dir_path, f'{file_prefix}.svg')
    plt.savefig(filepath, dpi=500, bbox_inches='tight')

    plt.close(fig)


def sort_and_remove_duplicates(arr):
    return sorted(list(set(arr)))

def gen_util_plot(parent_dir, algo, num_runs, iterations_to_save, normalize, x_max, y_max):
    print('-'*20)
    print(f'generating for algorithm: {algo}')
    cfg_file = get_cfg_dir(parent_dir, algo)

    setting_idx = 0
    print(f'Iterations to save: {iterations_to_save}')

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    util_save_every_nth_iteration = params['util_save_every_nth_iteration']
    iterations_to_save = [i // util_save_every_nth_iteration for i in iterations_to_save]
    iterations_to_save = sort_and_remove_duplicates(iterations_to_save)
    print(f'Util save every nth iteration: {util_save_every_nth_iteration}')
    print(f'Iterations to save (divided): {iterations_to_save}')

    print("Parent dir: ", parent_dir)

    plot_save_dir = params['data_dir'].replace("data", "utils_plots")
    plot_save_dir = os.path.join(parent_dir, plot_save_dir)
    os.makedirs(plot_save_dir, exist_ok=True)

    print("Plot save dir: ", plot_save_dir)

    util_data_all = []
    was_skipped = False

    for idx in range(num_runs):
        util_save_dir = params['data_dir'].replace("data", "utils_saved")
        util_save_dir = os.path.join(parent_dir, util_save_dir, str(setting_idx), str(idx))
        util_save_file = os.path.join(util_save_dir, 'util')
        print(f'Loading data from {util_save_file}')
        
        # util_data_all = append_util_data(util_data_all, util_save_file, iterations_to_save)
        if os.path.exists(util_save_file) is False:
            print(f'File {util_save_file} does not exist. Skipping run {idx}.')
            was_skipped = True
            continue

        with open(util_save_file, 'rb') as f:
            util_data = pickle.load(f)

        util_data = np.array([[t.numpy() for t in util_data[i]] for i in range(len(util_data))])
        util_data_all.append(util_data)

    if was_skipped:
        print(f'Skipped some, quiting.')
        quit(0)

    for iteration_id in iterations_to_save:
        true_iteration_id = iteration_id * util_save_every_nth_iteration
        dividy_by = num_runs

        chosen_util_data = []
        for run_id in range(num_runs):
            chosen_util_data.append(util_data_all[run_id][iteration_id])

        file_pref = f'util_iteration={true_iteration_id}'

        # Create histograms for this iteration with all results among the runs.
        create_histogram(chosen_util_data, 
                        dir_path=plot_save_dir, 
                        file_prefix=file_pref,
                        title=f'Utils at iteration {true_iteration_id}, normalize: {normalize}', 
                        normalize=normalize,
                        divide_average_by=dividy_by,
                        x_max=x_max, y_max=y_max)
        
        print(f'Saved plots and data to {plot_save_dir}, f: {file_pref}')


def get_cfg_dir(parent_dir, algo):
    return parent_dir + f"/cfg/{algo}.json"



def func_take_nth_max(n):
    def nth_max(arr):
        copy_array = arr.copy()
        sorted_arr = np.sort(copy_array)[::-1]
        return sorted_arr[n-1]
    return nth_max




def add_cfg_performance_util(parent_dir, cfg, iterations_to_save, setting_idx, num_runs, normalize, func, m):
    with open(cfg, 'r') as f:
        params = json.load(f)

    util_save_every_nth_iteration = params['util_save_every_nth_iteration']

    iterations_to_save = [i // util_save_every_nth_iteration for i in iterations_to_save]
    iterations_to_save = sort_and_remove_duplicates(iterations_to_save)

    averages = []

    for idx in range(num_runs):
        util_save_dir = params['data_dir'].replace("data", "utils_saved")
        util_save_dir = os.path.join(parent_dir, util_save_dir, str(setting_idx), str(idx))
        util_save_file = os.path.join(util_save_dir, 'util')
        # print(f'Loading data from {util_save_file} and {bias_corrected_util_save_file}')
        
        with open(util_save_file, 'rb') as f:
            util_data = pickle.load(f)

        util_data = np.array([[t.numpy() for t in util_data[i]] for i in range(len(util_data))])

        # print(f'cfg: {cfg}, setting_idx: {setting_idx}')
        # print(f'   util dadta shape: {util_data.shape}')
        # quit(1)

        cur_averages = []
        for iter_id in range(len(iterations_to_save)):
            cur_data = np.array(util_data[iterations_to_save[iter_id]])
            if normalize:
                cur_data = np.array([normalize_array(arr) for arr in cur_data])
            cur_data = cur_data.flatten()

            average_value = func(cur_data)
            cur_averages.append(average_value)
            # print(f'idx: {idx}, iter_id: {iter_id}, cur_data: {cur_data}, average_value: {average_value}')
            # averages[idx][iter_id] = average_value

        averages.append(bin_m_errs_np_arr(errs=np.array(cur_averages), m=m))
        # averages.append(np.array(cur_averages))

        print(f' finished plotting cfg: {cfg}, setting: {setting_idx}, run: {idx}')

    return np.array(averages)
    
def generate_util_maxes_plot_for_algo(parent_dir, algo, num_runs, normalize, m, iterations_to_save, xticks, xticks_labels):
    print('-'*20)
    print(f'Util maxes plot for algorithm: {algo}')

    label = get_label(algo)
    labels = [f'{label} 1st max', f'{label} 2nd max', f'{label} 3rd max', f'{label} 4th max', f'{label} 5th max']
    performances = []
    cfg_dir = get_cfg_dir(parent_dir, algo)

    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_take_nth_max(1), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_take_nth_max(2), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_take_nth_max(3), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_take_nth_max(4), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_take_nth_max(5), m=m
    ))

    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=16,
        filename_pref=f'{algo}_util_maxes_runs={num_runs}',
        svg=True,
    )


def generate_util_plot_for_all_algos_for_func(parent_dir, num_runs, iterations_to_save, normalize, func, file_pref, m, xticks, xticks_labels):
    print('-'*20)
    print(f'generating plots for {file_pref} with all algos')
    labels = []
    performances = []
    colors = []

    for algo in all_algos():
        labels.append(get_label(algo))
        colors.append(get_color(algo))

        performances.append(add_cfg_performance_util(
            parent_dir=parent_dir, cfg=get_cfg_dir(parent_dir, algo), 
            iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
            normalize=normalize, func=func, m=m
        ))
    

    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=16,
        filename_pref=f'{file_pref}_{algo}_runs={num_runs}',
        svg=True,
    )


def generate_mean_plots_for_all_algos(parent_dir, num_runs, iterations_to_save, normalize, m, xticks, xticks_labels):
    generate_util_plot_for_all_algos_for_func(parent_dir, num_runs, iterations_to_save, normalize, 
                                              np.mean, "util_mean", m, xticks, xticks_labels)
    


def generate_max_plots_for_all_algos(parent_dir, num_runs, iterations_to_save, normalize, m, xticks, xticks_labels):
    generate_util_plot_for_all_algos_for_func(parent_dir, num_runs, iterations_to_save, normalize, 
                                              np.max, "util_max", m, xticks, xticks_labels)
    

def all_algos():
    return ['bp', 'l2', 'snp', 'cbp', 'cbp_l2', 'cbp_snp']