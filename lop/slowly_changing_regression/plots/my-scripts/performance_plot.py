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

    # parent_dir1 = "/home/aldas/TUDelft/RP/results_copied/05-22_result-backup/slowly_flip-one-false_5runs"
    # parent_dir2 = "/home/aldas/TUDelft/RP/results_copied/05-23_DAIC_slowly-5runs"
    # parent_dir3 = "/home/aldas/TUDelft/RP/results_copied/05-25_DAIC_original-codebase-snp_5runs"
    # parent_dir4 = "/home/aldas/TUDelft/RP/results_copied/06-04_slowly-l2-snp-hyperparams"
    # cfg_file_bp, bp_settings_idx = (parent_dir1 + "/cfg/sgd/bp/relu.json"), 2
    # cfg_file_cbp, cbp_settings_idx = (parent_dir1 + "/cfg/sgd/cbp/relu.json") , 2
    # cfg_file_l2, l2_settings_idx = (parent_dir4 + "/cfg/l2.json"), 1
    # cfg_file_snp, snp_settings_idx = (parent_dir4 + "/cfg/snp.json"), 3
    # cfg_file_cbp_snp, cbp_snp_settings_idx = (parent_dir4 + "/cfg/cbp_snp.json"), 25
    # cfg_file_cbp_l2, cbp_l2_setting_idx = (parent_dir4 + "/cfg/cbp_with_l2.json"), 12

    parent_dir_util_dropout = "/home/aldas/TUDelft/RP/results_copied/06-09_dropout_test"
    cfg_file_util_dropout = parent_dir_util_dropout + "/cfg/util_dropout.json"

    with open(cfg_file_util_dropout, 'r') as f:
        params = json.load(f)

    performances = []
    # m = int(params['flip_after'])*5*3
    m = 10000*5*3

    h, param_settings = get_configurations(params=params)
    # print(f'h:{h}, param_settings: {param_settings[25]}')
    print(f'param_settings len: {len(param_settings)}')
    # selected_indeces = [12, 21,22,23,24]
    # labels = [param_settings[ind] for ind in selected_indeces]
    labels = param_settings
    num_runs = 5
    
    # labels=['CBP', 'SnP', 'CBP+SnP', 'CBP+L2']
    # labels=['BP', 'CBP', 'SnP', 'L2', 'CBP+SnP', 'CBP+L2']
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir1, cfg=cfg_file_bp, setting_idx=bp_settings_idx, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir1, cfg=cfg_file_cbp, setting_idx=cbp_settings_idx, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir4, cfg=cfg_file_snp, setting_idx=snp_settings_idx, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir4, cfg=cfg_file_l2, setting_idx=l2_settings_idx, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir4, cfg=cfg_file_cbp_snp, setting_idx=cbp_snp_settings_idx, m=m, num_runs=num_runs))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir4, cfg=cfg_file_cbp_l2, setting_idx=cbp_l2_setting_idx, m=m, num_runs=num_runs))
    
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
        # yticks=yticks,
        # xticks=[0, 500000, 1000000],
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,  # divided by dead neuron measure period
        labels=labels,
        fontsize=18,
        filename_pref='util_dropout_error',
        svg=True,
    )


if __name__ == '__main__':
    sys.exit(main())

