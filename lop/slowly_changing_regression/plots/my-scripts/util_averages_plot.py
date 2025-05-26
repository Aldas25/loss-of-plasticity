import sys
import json
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot

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

def create_histogram(util_data, dir_path='plots', file_prefix='non-normalized_util_data', title='title', normalize=False, divide_average_by=1):
    # Prepare data
    data = np.array(util_data) # Assuming util_data is a list of numpy arrays
    # print(f'{title}, data: {data[:20]}')
    # data = np.array([t.numpy() for t in util_data])
    if normalize:
        data = np.array([normalize_array(arr) for arr in data])
    data = data.flatten()
    print(f'{title}, data: {data[:20]}')

    # plt.close('all') # in case some other plot is open

    fig, ax = plt.subplots(figsize=(10, 6))
    num_bins = 20
    weights = np.ones_like(data) / divide_average_by if divide_average_by > 1 else None
    hist, bins, patches = ax.hist(data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7, weights=weights)
    
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    # ax.axvline(np.mean(util_data), color='red', linestyle='dashed', linewidth=2, label='Mean')
    
    if divide_average_by > 1:
        title = f'{title} (Averaged over {divide_average_by} runs)'
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    # plt.show()

    filepath = os.path.join(dir_path, f'{file_prefix}.png')
    plt.savefig(filepath, dpi=500, bbox_inches='tight')

    plt.close(fig)

    # Calculate median, average, and standard deviation
    median_value = np.median(data)
    average_value = np.mean(data)
    std_value = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)

    # Prepare the statistics output file path
    stats_file_path = os.path.join(dir_path, f'{file_prefix}_stats.txt')

    # Write the statistics to the file
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write(f"Median: {median_value}\n")
        stats_file.write(f"Average: {average_value}\n")
        stats_file.write(f"Standard Deviation: {std_value}\n")
        stats_file.write(f"Min: {min_value}\n")
        stats_file.write(f"Max: {max_value}\n")


def sort_and_remove_duplicates(arr):
    return sorted(list(set(arr)))


def add_cfg_performance(parent_dir, cfg, iterations_to_save, setting_idx, num_runs, normalize=False, func=np.mean):
    with open(cfg, 'r') as f:
        params = json.load(f)

    util_save_every_nth_iteration = params['util_save_every_nth_iteration']

    iterations_to_save = [i // util_save_every_nth_iteration for i in iterations_to_save]
    iterations_to_save = sort_and_remove_duplicates(iterations_to_save)

    m = int(params['flip_after'])*2

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

        averages.append(bin_m_errs_np_arr(errs=np.array(cur_averages), m=100))
        # averages.append(np.array(cur_averages))

        print(f' finished plotting cfg: {cfg}, setting: {setting_idx}, run: {idx}')

    return np.array(averages)
    
def func_take_nth_max(n):
    def nth_max(arr):
        copy_array = arr.copy()
        sorted_arr = np.sort(copy_array)[::-1]
        return sorted_arr[n-1]
    return nth_max
    
def main():
    parent_dir1 = "/home/aldas/TUDelft/RP/results_copied/05-22_result-backup/slowly_flip-one-false_5runs"
    parent_dir2 = "/home/aldas/TUDelft/RP/results_copied/05-23_DAIC_slowly-5runs"
    parent_dir3 = "/home/aldas/TUDelft/RP/results_copied/05-25_DAIC_original-codebase-snp_5runs"
    cfg_file_bp = parent_dir1 + "/cfg/sgd/bp/relu.json" 
    cfg_file_cbp = parent_dir1 + "/cfg/sgd/cbp/relu.json" 
    cfg_file_l2 = parent_dir2 + "/cfg/sgd/l2/l2.json"
    cfg_file_cbp_l2 = parent_dir2 + "/cfg/sgd/cbp/cbp_with_l2.json"
    cfg_file_snp = parent_dir3 + "/cfg/sgd/shrink-and-perturb/snp.json"
    cfg_file_cbp_snp = parent_dir3 + "/cfg/sgd/shrink-and-perturb/cbp_snp.json"
    num_runs = 5

    labels = ['CBP+SnP 1st max', 'CBP+SnP 2nd max', 'CBP+SnP 3rd max', 'CBP+SnP 4th max', 'CBP+SnP 5th max']
    performances = []

    # iterations_to_save=list(range(int(1e6), int(1e6)+int(1e5)))
    iterations_to_save=list(range(int(3e6)))
    # iterations_to_save=list(range(int(1e5)))
    # iterations_to_save=list(range(int(3e6)-int(1e5), int(3e6)))

    performances.append(add_cfg_performance(
        parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
        normalize=False, func=func_take_nth_max(1)
    ))
    performances.append(add_cfg_performance(
        parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
        normalize=False, func=func_take_nth_max(2)
    ))
    performances.append(add_cfg_performance(
        parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
        normalize=False, func=func_take_nth_max(3)
    ))
    performances.append(add_cfg_performance(
        parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
        normalize=False, func=func_take_nth_max(4)
    ))
    performances.append(add_cfg_performance(
        parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
        normalize=False, func=func_take_nth_max(5)
    ))

    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir1, cfg=cfg_file_bp, iterations_to_save=iterations_to_save, setting_idx=2, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir1, cfg=cfg_file_cbp, iterations_to_save=iterations_to_save, setting_idx=2, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir3, cfg=cfg_file_snp, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir2, cfg=cfg_file_l2, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, iterations_to_save=iterations_to_save, setting_idx=10, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir2, cfg=cfg_file_cbp_l2, iterations_to_save=iterations_to_save, setting_idx=7, num_runs=num_runs, 
    #     normalize=False, func=np.mean
    # ))


    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        # yticks=yticks,
        # xticks=[0, 1500000, 3000000], 
        # xticks_labels=['0', '1.5M', '3M'],
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=100 * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=16,
        filename_pref='cbp_snp_util_variuos_max'
    )

           

if __name__ == '__main__':
    sys.exit(main())
