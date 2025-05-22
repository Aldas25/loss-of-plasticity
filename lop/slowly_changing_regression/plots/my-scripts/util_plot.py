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


def append_util_data(util_data_all, util_save_file, iterations_to_save):
    with open(util_save_file, 'rb') as f:
        util_data_from_file = pickle.load(f)

    # print(f'len of util data: {len(util_data)}')
    # print(f'Type of util_data: {type(util_data)}')

    for (idx, iteration_id) in enumerate(iterations_to_save):
        # print(f'idx: {idx}, iteration_id: {iteration_id}')
        util_data = np.array([t.numpy() for t in util_data_from_file[iteration_id]])
        for i in range(max(len(util_data_all[idx]), len(util_data))):
            if i >= len(util_data_all):
                util_data_all[idx].append([])
            
            if i < len(util_data):
                util_data_all[idx][i].extend(util_data[i])

    return util_data_all

def sort_and_remove_duplicates(arr):
    return sorted(list(set(arr)))

def main():
    cfg_file = "/scratch/alenksas/results/05-18_slowly_10_runs_flip-one-false/cfg/sgd/shrink-and-perturb/snp.json"  # left: cbp snp l2
    iterations_to_save = [int(3e6 - 1)]
    iterations_to_save.extend([int(i * 1e5) for i in range(30)])
    iterations_to_save.extend([int(3e6-1e4 + i*1e3) for i in range(10)])
    print(f'Iterations to save: {iterations_to_save}')

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    util_save_every_nth_iteration = params['util_save_every_nth_iteration']
    iterations_to_save = [i // util_save_every_nth_iteration for i in iterations_to_save]
    iterations_to_save = sort_and_remove_duplicates(iterations_to_save)
    print(f'Util save every nth iteration: {util_save_every_nth_iteration}')
    print(f'Iterations to save (divided): {iterations_to_save}')

    parent_dir = "/scratch/alenksas/results/05-18_slowly_10_runs_flip-one-false"
    print("Parent dir: ", parent_dir)

    plot_save_dir = params['data_dir'].replace("data", "utils_plots")
    plot_save_dir = os.path.join(parent_dir, plot_save_dir)
    os.makedirs(plot_save_dir, exist_ok=True)

    print("Plot save dir: ", plot_save_dir)

    m = int(params['flip_after'])*2

    param_settings_names, param_settings = get_configurations(params=params)
    num_runs = params['num_runs']

    for setting_idx in range(len(param_settings)):
        hidden_layer_cnt = 1 # 1 is hardcoded for now
        util_data_all = [[[] for _ in range(hidden_layer_cnt)] for _ in range(len(iterations_to_save))]
        bias_corrected_util_data_all = [[[] for _ in range(hidden_layer_cnt)] for _ in range(len(iterations_to_save))]

        for idx in range(num_runs):
            util_save_dir = params['data_dir'].replace("data", "utils_saved")
            util_save_dir = os.path.join(parent_dir, util_save_dir, str(setting_idx), str(idx))
            util_save_file = os.path.join(util_save_dir, 'util')
            bias_corrected_util_save_file = os.path.join(util_save_dir, 'bias_corrected_util')
            print(f'Loading data from {util_save_file} and {bias_corrected_util_save_file}')
            
            util_data_all = append_util_data(util_data_all, util_save_file, iterations_to_save)
            bias_corrected_util_data_all = append_util_data(bias_corrected_util_data_all, bias_corrected_util_save_file, iterations_to_save)

            # print(f'setting_idx: {setting_idx}, idx: {idx}, data size: {len(util_data)}, {len(bias_corrected_util_data)}')
            # print(f'Util data: {util_data[:20]}')
            # print(f'Bias corrected util data: {bias_corrected_util_data[:20]}')

        print(f'read data. util shape: {np.array(util_data_all).shape}, bias_corrected_util shape: {np.array(bias_corrected_util_data_all).shape}')

        for (iter_idx, iteration_id) in enumerate(iterations_to_save):
            true_iteration_id = iteration_id * util_save_every_nth_iteration
            dividy_by = num_runs

            cur_plot_save_dir = os.path.join(
                                plot_save_dir, f'{param_settings_names[0]}={param_settings[setting_idx][0]}', f'iteration={true_iteration_id}'
                                )
            os.makedirs(cur_plot_save_dir, exist_ok=True)
            print(f'cur_plot_save_dir: {cur_plot_save_dir}')

            # print(f'{param_settings}, cur: {param_settings[setting_idx][0]}')

            # Create histograms for this iteration with all results among the runs.
            create_histogram(util_data_all[iter_idx], 
                            dir_path=cur_plot_save_dir, 
                            file_prefix=f'util',
                            title=f'Util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {true_iteration_id}', normalize=False,
                            divide_average_by=dividy_by)
            
            create_histogram(util_data_all[iter_idx], 
                            dir_path=cur_plot_save_dir, 
                            file_prefix=f'util_normalized',
                            title=f'Normalized util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {true_iteration_id}', normalize=True,
                            divide_average_by=dividy_by)
            
            create_histogram(bias_corrected_util_data_all[iter_idx], 
                            dir_path=cur_plot_save_dir, 
                            file_prefix=f'bias_corrected_util',
                            title=f'Bias corrected util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {true_iteration_id}', normalize=False,
                            divide_average_by=dividy_by)
            
            create_histogram(bias_corrected_util_data_all[iter_idx], 
                            dir_path=cur_plot_save_dir, 
                            file_prefix=f'bias_corrected_util_normalized',
                            title=f'Normalized bias corrected util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {true_iteration_id}', normalize=True,
                            divide_average_by=dividy_by)

            print(f'Saved plots and data to {cur_plot_save_dir}')
           

if __name__ == '__main__':
    sys.exit(main())
