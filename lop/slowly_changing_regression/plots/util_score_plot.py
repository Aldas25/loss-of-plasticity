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
    """Normalize a single array to the range [-1, 1]"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Check if the array has a non-zero range to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(arr)  # Return zeros if array is constant
    
    # Scale to [-1, 1]
    return 2 * (arr - min_val) / (max_val - min_val) - 1

def create_histogram(util_data, dir_path='plots', file_prefix='non-normalized_util_data', title='title', normalize=False):
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
    hist, bins, patches = ax.hist(data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    # ax.axvline(np.mean(util_data), color='red', linestyle='dashed', linewidth=2, label='Mean')

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

    # Prepare the statistics output file path
    stats_file_path = os.path.join(dir_path, f'{file_prefix}_stats.txt')

    # Write the statistics to the file
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write(f"Median: {median_value}\n")
        stats_file.write(f"Average: {average_value}\n")
        stats_file.write(f"Standard Deviation: {std_value}\n")


def append_util_data(util_data_all, util_save_file, iteration_id=0):
    with open(util_save_file, 'rb') as f:
        util_data = pickle.load(f)

    util_data = np.array([t.numpy() for t in util_data[iteration_id]])
    for i in range(max(len(util_data_all), len(util_data))):
        if i >= len(util_data_all):
            util_data_all.append([])
        
        if i < len(util_data):
            util_data_all[i].extend(util_data[i])

    return util_data_all

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # change the cfg file to get the results for different activation functions, ex. '../cfg/sgd/bp/tanh.json'
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/sgd/bp/relu.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    plot_save_dir = params['data_dir'].replace("data", "utils_plots")
    plot_save_dir = os.path.join(os.path.pardir, plot_save_dir)
    os.makedirs(plot_save_dir, exist_ok=True)

    m = int(params['flip_after'])*2

    iteration_id = 10000 - 1  # the iteration which will have the plot saved

    param_settings_names, param_settings = get_configurations(params=params)
    # labels = param_settings
    num_runs = params['num_runs']
    indices = [i for i in range(len(param_settings))]
    for setting_idx in indices:
        hidden_layer_cnt = 1 # 1 is hardcoded for now
        util_data_all = [[] for _ in range(hidden_layer_cnt)]
        bias_corrected_util_data_all = [[] for _ in range(hidden_layer_cnt)]

        for idx in range(num_runs):
            util_save_dir = params['data_dir'].replace("data", "utils_saved")
            util_save_dir = os.path.join(os.path.pardir, util_save_dir, str(setting_idx), str(idx))
            util_save_file = os.path.join(util_save_dir, 'util')
            bias_corrected_util_save_file = os.path.join(util_save_dir, 'bias_corrected_util')
            
            util_data_all = append_util_data(util_data_all, util_save_file, iteration_id=iteration_id)
            bias_corrected_util_data_all = append_util_data(bias_corrected_util_data_all, bias_corrected_util_save_file, iteration_id=iteration_id)

            # print(f'setting_idx: {setting_idx}, idx: {idx}, data size: {len(util_data)}, {len(bias_corrected_util_data)}')
            # print(f'Util data: {util_data[:20]}')
            # print(f'Bias corrected util data: {bias_corrected_util_data[:20]}')

        cur_plot_save_dir = os.path.join(
                             plot_save_dir, f'{param_settings_names[0]}={param_settings[setting_idx][0]}', f'iteration={iteration_id}'
                             )
        os.makedirs(cur_plot_save_dir, exist_ok=True)

        # print(f'{param_settings}, cur: {param_settings[setting_idx][0]}')

        # Create histograms for this iteration with all results among the runs.
        create_histogram(util_data_all, 
                         dir_path=cur_plot_save_dir, 
                         file_prefix=f'util',
                         title=f'Util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {iteration_id}', normalize=False)
        
        create_histogram(util_data_all, 
                         dir_path=cur_plot_save_dir, 
                         file_prefix=f'util_normalized',
                         title=f'Normalized util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {iteration_id}', normalize=True)
        
        create_histogram(bias_corrected_util_data_all, 
                         dir_path=cur_plot_save_dir, 
                         file_prefix=f'bias_corrected_util',
                         title=f'Bias corrected util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {iteration_id}', normalize=False)
        
        create_histogram(bias_corrected_util_data_all, 
                         dir_path=cur_plot_save_dir, 
                         file_prefix=f'bias_corrected_util_normalized',
                         title=f'Normalized bias corrected util data for {param_settings_names[0]}={param_settings[setting_idx][0]} at iteration {iteration_id}', normalize=True)

        print(f'Saved plots and data to {cur_plot_save_dir}')
           

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

