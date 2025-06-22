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
    parent_dir = "..." # fill in the folder path
    num_runs = 10
    normalize = False

    one_task = 10*1000
    total = int(3e6)
    tasks = total // one_task

    iterations_to_save = [total - 1]  # very last one
    iterations_to_save.extend([i * one_task * 10 + one_task-1 for i in range(29)])  # some last iterations of each task
    iterations_to_save.extend([total - one_task + i*(one_task//10) for i in range(10)]) # check how changes in the last task
    iterations_to_save.extend([i*(one_task//10) for i in range(10)]) # check how changes in the first task
    iterations_to_save.extend([(tasks // 2)*one_task + i*(one_task//10) for i in range(10)]) # check how changes in some middle task
    

    for algo in all_algos():
        gen_util_plot(parent_dir=parent_dir, algo=algo, num_runs=num_runs, iterations_to_save=iterations_to_save, 
                      normalize=normalize, x_max=5.0, y_max=5.0)


if __name__ == '__main__':
    sys.exit(main())
