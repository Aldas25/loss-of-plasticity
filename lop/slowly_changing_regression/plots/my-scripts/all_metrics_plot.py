import sys
import json
import pickle
import os
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(parent_dir, cfg='', setting_idx=0, m=2*10*1000, num_runs=30, metric='unknown'):
    if metric == 'unknown':
        print('Please specify the metric to be used for performance calculation.')
        quit(0)

    with open(cfg, 'r') as f:
        params = json.load(f)
    # list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = parent_dir + '/' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if metric == 'error':
            # Online performance
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))
        elif metric == 'dead_neurons':
            # Dead neurons
            neurons_max = 5
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'] / neurons_max, m=m)))
        elif metric == 'weights':
            # Weights
            num_weights = 105
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight'] / num_weights, m=m)))
        else:
            print(f'Unknown metric: {metric}')
            quit(0)

    # print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def generate_plot_for_metric(parent_dir, num_runs, metric):
    print(f'Plotting metric: {metric}')
    # add plot for all algorithms

    m = {'error': 10000*5*2, 'dead_neurons': 10000*5*2, 'weights': 10000*5*2}[metric]
    
    labels = []
    performances = []
    colors = []

    for algo in all_algos():
        print(f'Adding performance for {algo} with metric {metric} and m={m}')
        labels.append(get_label(algo))
        colors.append(get_color(algo))

        cfg_file = get_cfg_dir(parent_dir, algo)
        performances.append(add_cfg_performance(
            parent_dir=parent_dir, cfg=cfg_file, m=m, num_runs=num_runs, metric=metric
        )) 

    performances = np.array(performances)

    generate_online_performance_plot(
        performances=performances,
        colors=colors,
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,
        labels=labels,
        fontsize=18,
        filename_pref=f'slowly_all_{metric}_runs={num_runs}',
        svg=True,
    )


def main():

    parent_dir = "..." # fill in the folder path

    num_runs = 10  # should be 100
    for metric in ['error', 'dead_neurons', 'weights']:
        generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric)

if __name__ == '__main__':
    sys.exit(main())

