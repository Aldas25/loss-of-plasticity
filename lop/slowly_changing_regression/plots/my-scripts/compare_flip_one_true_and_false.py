import sys
import json
import pickle
import argparse
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
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=3*m)))

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def main(arguments):
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # change the cfg file to get the results for different activation functions, ex. '../cfg/sgd/bp/tanh.json'
    # parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str,
                            # default='../cfg/sgd/bp/relu.json')
    # args = parser.parse_args(arguments)
    # cfg_file = args.c

    parent_dir1 = "/scratch/alenksas/results/slowly_flip-one-true_5runs"
    parent_dir2 = "/scratch/alenksas/results/slowly_flip-one-false_10runs"
    cfg_suf = "/cfg/sgd/bp/relu.json" 

    cfg_file1 = parent_dir1 + cfg_suf
    cfg_file2 = parent_dir2 + cfg_suf

    with open(cfg_file1, 'r') as f:
        params = json.load(f)

    performances = []
    m = int(params['flip_after'])*2

    _, param_settings = get_configurations(params=params)
    # labels = param_settings
    labels = ['flip_one=true', 'flip_one=false']
    # num_runs = params['num_runs']
    num_runs = 5
    performances.append(add_cfg_performance(parent_dir=parent_dir1, cfg=cfg_file1, setting_idx=2, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir2, cfg=cfg_file2, setting_idx=2, m=m, num_runs=num_runs))
    # for i in range(len(param_settings)):
    #     performances.append(add_cfg_performance(parent_dir=parent_dir, cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/bp/linear.json', setting_idx=0, m=m, num_runs=num_runs))
    # labels.append('linear')
    performances = np.array(performances)

    # if params['hidden_activation'] in ['relu', 'swish', 'leaky_relu']:
    #     yticks = [0.6, 0.8, 1., 1.2, 1.4]
    # else:
    #     yticks = [0.4, 0.6, 0.8, 1, 1.2]

    yticks = [0.4, 0.6, 0.8, 1., 1.2, 1.4]

    # print(yticks, params['hidden_activation'])
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9'],
        yticks=yticks,
        xticks=[0, 500000, 1000000],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,
        labels=labels
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

