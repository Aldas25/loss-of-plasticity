import sys
import json
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def main():
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    num_runs = 10
    normalize = False
    m = 300
    # m = 1

    xticks=[0, int(0.5 * 3e6), int(3e6)]
    # xticks=[0, int(1e4)]
    xticks_labels=['0', '1.5M', '3M']
    # xticks_labels=['0', '3M-1e4']

    iterations_to_save=list(range(int(3e6)))
    # iterations_to_save=list(range(int(1e6), int(1e6)+int(1e5)))
    # iterations_to_save=list(range(int(1e5)))
    # iterations_to_save=list(range(int(3e6)-int(1e4), int(3e6)))

    # for algo in all_algos():
        # generate_util_maxes_plot_for_algo(parent_dir, algo, num_runs, normalize, m, iterations_to_save, xticks, xticks_labels)

    generate_mean_plots_for_all_algos(parent_dir, num_runs, iterations_to_save, normalize, m, xticks, xticks_labels)
    # generate_max_plots_for_all_algos(parent_dir, num_runs, iterations_to_save, normalize, m, xticks, xticks_labels)
           

if __name__ == '__main__':
    sys.exit(main())
