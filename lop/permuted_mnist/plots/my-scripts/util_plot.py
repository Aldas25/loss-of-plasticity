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
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-07_permuted-30runs"

    one_task = 10*1000
    num_tasks = 800
    normalize = True
    total = num_tasks * one_task

    iterations_to_save = [int(total - 1)]  # very last one
    iterations_to_save.extend([int(80 * i * one_task)+one_task - 1 for i in range(num_tasks // 80)])  # some last iterations of some tasks
    iterations_to_save.extend([int(total - one_task + i*(one_task // 10)) for i in range(10)]) # check how changes in the last task
    iterations_to_save.extend([int(i*(one_task // 10)) for i in range(10)]) # check how changes in the first task
    iterations_to_save.extend([int((total // 2) + i*(one_task // 10)) for i in range(10)]) # check how changes in some middle task

    num_runs = 4

    for algo in all_algos():
        gen_util_plot(parent_dir=parent_dir, algo=algo, num_runs=num_runs, iterations_to_save=iterations_to_save, 
                      normalize=normalize, x_max=1.0, y_max=220.0)


if __name__ == '__main__':
    sys.exit(main())
