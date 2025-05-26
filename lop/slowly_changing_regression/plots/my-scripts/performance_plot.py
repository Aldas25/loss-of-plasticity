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
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))

        # print(f'data: {data}')

        # Dead neurons
        # dead_neurons_measure_period = 100
        # neurons_max = 5
        # per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'] / neurons_max, m=m // dead_neurons_measure_period)))
        # per_param_setting_performance.append(np.array(data['dead_neurons'] / neurons_max))

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def main(arguments):
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # change the cfg file to get the results for different activation functions, ex. '../cfg/sgd/bp/tanh.json'
    # parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str,
                            # default='../cfg/sgd/bp/relu.json')
    # args = parser.parse_args(arguments)
    # cfg_file = args.c

    parent_dir1 = "/home/aldas/TUDelft/RP/results_copied/05-22_result-backup/slowly_flip-one-false_5runs"
    parent_dir2 = "/home/aldas/TUDelft/RP/results_copied/05-23_DAIC_slowly-5runs"
    parent_dir3 = "/home/aldas/TUDelft/RP/results_copied/05-25_DAIC_original-codebase-snp_5runs"
    cfg_file_bp = parent_dir1 + "/cfg/sgd/bp/relu.json" 
    cfg_file_cbp = parent_dir1 + "/cfg/sgd/cbp/relu.json" 
    cfg_file_l2 = parent_dir2 + "/cfg/sgd/l2/l2.json"
    cfg_file_cbp_l2 = parent_dir2 + "/cfg/sgd/cbp/cbp_with_l2.json"
    cfg_file_snp = parent_dir3 + "/cfg/sgd/shrink-and-perturb/snp.json"
    cfg_file_cbp_snp = parent_dir3 + "/cfg/sgd/shrink-and-perturb/cbp_snp.json"

    with open(cfg_file_cbp_snp, 'r') as f:
        params = json.load(f)

    performances = []
    m = int(params['flip_after'])*5*3

    _, param_settings = get_configurations(params=params)
    # selected_indeces = [4, 10, 18]
    # labels = [param_settings[ind] for ind in selected_indeces]
    # print(f'len of params: {len(param_settings)}')
    # labels = param_settings
    # labels = ['Continual BP', 'Backprop']
    # num_runs = params['num_runs']
    num_runs = 5
    
    labels=['BP', 'CBP', 'SnP', 'L2', 'CBP+SnP', 'CBP+L2']
    performances.append(add_cfg_performance(parent_dir=parent_dir1, cfg=cfg_file_bp, setting_idx=2, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir1, cfg=cfg_file_cbp, setting_idx=2, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir3, cfg=cfg_file_snp, setting_idx=0, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir2, cfg=cfg_file_l2, setting_idx=0, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir3, cfg=cfg_file_cbp_snp, setting_idx=10, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(parent_dir=parent_dir2, cfg=cfg_file_cbp_l2, setting_idx=7, m=m, num_runs=num_runs))
    
    # for i in selected_indeces:
    # for i in range(len(param_settings)):
    #     performances.append(add_cfg_performance(parent_dir=parent_dir, cfg=cfg_file_cbp_snp, setting_idx=i, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/bp/linear.json', setting_idx=0, m=m, num_runs=num_runs))
    # labels.append('linear')
    performances = np.array(performances)

    # if params['hidden_activation'] in ['relu', 'swish', 'leaky_relu']:
    #     yticks = [0.6, 0.8, 1., 1.2, 1.4]
    # else:
    #     yticks = [0.4, 0.6, 0.8, 1, 1.2]

    # yticks = [0.4, 0.6, 0.8, 1., 1.2, 1.4]

    # print(yticks, params['hidden_activation'])
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0', 'C1', 'C2', 'C6', 'C7']*5,
        # yticks=yticks,
        # xticks=[0, 500000, 1000000],
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,  # divided by dead neuron measure period
        labels=labels,
        fontsize=18,
        filename_pref='asdasd'
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

