import sys
import json
import pickle
import argparse
import os
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        parent_dir = "/scratch/alenksas/results/05-18_slowly_10_runs_flip-one-true"
        file = parent_dir + '/' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=2*m)))

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def main(arguments):
    cfg_file = "/.../snp.json"  # fill in the file path

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    performances = []
    m = int(params['flip_after'])*2

    _, param_settings = get_configurations(params=params)
    labels = param_settings
    num_runs = params['num_runs']
    for i in range(len(param_settings)):
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))
    performances = np.array(performances)

    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9'],
        # yticks=yticks,
        xticks=[0, 750000, 1500000],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,
        labels=labels
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

