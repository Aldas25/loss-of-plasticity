import sys
import json
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


    
def func_average_maximums_from_range(from_idx, to_idx):
    def average_maximums(arr):
        copy_array = arr.copy()
        sorted_arr = np.sort(copy_array)[::-1]
        return np.mean(sorted_arr[(from_idx-1):to_idx])
    return average_maximums


def mnist_generate_util_maxes_plot_for_algo(parent_dir, algo, num_runs, normalize, m, iterations_to_save, xticks, xticks_labels):
    print('-'*20)
    print(f'Util maxes plot for algorithm: {algo}')

    label = get_label(algo)
    labels = [f'{label} 1-50 max', f'{label} 51-100 max', f'{label} 101-150 max', f'{label} 151-200 max', f'{label} 201-250 max']
    performances = []
    cfg_dir = get_cfg_dir(parent_dir, algo)

    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_average_maximums_from_range(1, 50), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_average_maximums_from_range(51, 100), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_average_maximums_from_range(101, 150), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_average_maximums_from_range(151, 200), m=m
    ))
    performances.append(add_cfg_performance_util(
        parent_dir=parent_dir, cfg=cfg_dir, iterations_to_save=iterations_to_save, setting_idx=0, num_runs=num_runs, 
        normalize=normalize, func=func_average_maximums_from_range(201, 250), m=m
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


def main():
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-07_permuted-30runs"
    num_runs = 4  # should be 30
    num_inputs = 10*1000
    xticks=[0, 400*num_inputs, 800*num_inputs]
    xticks_labels=['0', '400', '800']
    m = 100
    normalize = True

    num_tasks = 800
    total_iterations = num_tasks * num_inputs

    iterations_to_save=list(range(total_iterations)) # all tasks
    # iterations_to_save = list(range(10*num_inputs)) # first 10 tasks
    # iterations_to_save = list(range(total_iterations - 10*num_inputs, total_iterations))  # last 10 tasks
    # iterations_to_save = list(range((num_tasks // 2) * num_inputs, (num_tasks // 2 + 10) * num_inputs))  # middle 10 tasks

    for algo in all_algos():
        mnist_generate_util_maxes_plot_for_algo(parent_dir, algo, num_runs, normalize, m, iterations_to_save, xticks, xticks_labels)

    generate_mean_plots_for_all_algos(parent_dir, num_runs, iterations_to_save, normalize, m, xticks, xticks_labels)
         

if __name__ == '__main__':
    sys.exit(main())
