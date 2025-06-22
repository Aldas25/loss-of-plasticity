import sys
import json
import pickle
import os
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(parent_dir, cfg='', setting_idx=0, m=2*10*1000, num_runs=30):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = parent_dir + '/' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))

        # print(f'data: {data}')

        # Dead neurons
        # dead_neurons_measure_period = 100
        # neurons_max = 5
        # per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'] / neurons_max, m=m // dead_neurons_measure_period)))
        # per_param_setting_performance.append(np.array(data['dead_neurons'] / neurons_max))

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def main():

    parent_dir_util_dropout = "..."  # fill in the folder path
    cfg_file_util_dropout = parent_dir_util_dropout + "/cfg/util_dropout.json"

    with open(cfg_file_util_dropout, 'r') as f:
        params = json.load(f)

    performances = []
    m = 10000*5*3

    h, param_settings = get_configurations(params=params)
    print(f'param_settings len: {len(param_settings)}')
    labels = param_settings
    num_runs = 5
    
    # for i in selected_indeces:
    for i in range(len(param_settings)):
        if i == 33 or i > 65: 
            continue
        to_append = add_cfg_performance(parent_dir=parent_dir_util_dropout, cfg=cfg_file_util_dropout, setting_idx=i, m=m, num_runs=num_runs)
        performances.append(to_append)
    
        print(f'appended shape for {i}: {np.array(to_append).shape}')
    performances = np.array(performances)

    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0', 'C1', 'C2', 'C6', 'C7']*50,
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,  
        labels=labels,
        fontsize=18,
        filename_pref='util_dropout_error',
        svg=True,
    )


if __name__ == '__main__':
    sys.exit(main())

