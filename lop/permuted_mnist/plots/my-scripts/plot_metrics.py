import sys
import json
import pickle
import argparse
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(parent_dir, cfg, setting_idx=0, m=2*10*1000, num_runs=30, metric='accuracy'):
    with open(cfg, 'r') as f:
       params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):

        file = parent_dir + '/' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if metric == 'weight':
            #num_weights = 9588000
            # l = 100 # hidden units per layer
            # num_weights = 10*l + 2*l*l + l*784

            num_weights = 99400
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight_mag_sum'].sum(dim=1)/num_weights, m=m)))
        elif metric == 'dead_neurons':
            #num_units = 3*2000
            num_units = 3*100
            # print(f'dead neuron shape: {np.array(data["dead_neurons"]).shape}')
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'].sum(dim=1)/num_units, m=m)))
        elif metric == 'effective_rank':
            rank_normlization = 3*2000/100
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['effective_ranks'].sum(dim=1)/rank_normlization, m=m)))
        else:
            # tmp = np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m))
            # per_param_setting_performance.append(tmp[200:])
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m)))

    per_param_np_arr = np.array(per_param_setting_performance)
    print(param_settings[setting_idx], setting_idx, per_param_np_arr.mean())
    return per_param_np_arr


def main(arguments):
    #parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
    #                        default='../cfg/bp/std_net.json')
    #parser.add_argument('--metric', help="Specify the metric you want to plot, the options are: accuracy, weight,"
    #                                     " dead_neurons, and effective_rank", type=str, default='accuracy')

    #args = parser.parse_args(arguments)
    #cfg_file = args.cfg_file
    #metric = args.metric

    num_runs = 5
    metric = 'accuracy'
    m = {'weight': 10*1000, 'accuracy': 10*1000, 'dead_neurons': 1}[metric]

    parent_dir = '/home/aldas/TUDelft/RP/results_copied/06-11_mnist-cbp_variantions_hyperparams_rerun'


    performances = []

    # bp_cfg, bp_setting_idx = parent_dir + '/cfg/bp/std_net.json', 2
    # l2_cfg, l2_setting_idx = parent_dir + '/cfg/l2.json', 0
    # snp_cfg, snp_setting_idx = parent_dir + '/cfg/snp.json', 1
    # cbp_cfg, cbp_setting_idx = parent_dir + '/cfg/cbp.json', 0
    # cbp_l2_cfg, cbp_l2_setting_idx = parent_dir + '/cfg/cbp_l2.json', 3
    cbp_snp_cfg, cbp_snp_setting_idx = parent_dir + '/cfg/cbp_snp.json', -1

    # labels = ['L2', 'SNP', 'CBP', 'CBP+L2', 'CBP+SNP']
    # labels = ['BP', 'L2', 'SNP', 'CBP', 'CBP+L2', 'CBP+SNP']

    with open(cbp_snp_cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    # print(f'CBP+L2 list_params: {list_params}, setting: {param_settings[cbp_l2_setting_idx]}')
    # quit(0)

    print(len(param_settings), 'param settings found')

    selected_ids = [20, 26]

    labels = [param_settings[i] for i in selected_ids]
    # labels = param_settings

    for i in selected_ids:
    # for i in range(len(labels)):
        performances.append(add_cfg_performance(
            parent_dir=parent_dir, cfg=cbp_snp_cfg, setting_idx=i, m=m, num_runs=num_runs, metric=metric))

    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=bp_cfg, setting_idx=bp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=l2_cfg, setting_idx=l2_setting_idx, m=m, num_runs=num_runs, metric=metric))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=snp_cfg, setting_idx=snp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=cbp_cfg, setting_idx=cbp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=cbp_l2_cfg, setting_idx=cbp_l2_setting_idx, m=m, num_runs=num_runs, metric=metric))
    # performances.append(add_cfg_performance(
    #     parent_dir=parent_dir, cfg=cbp_snp_cfg, setting_idx=cbp_snp_setting_idx, m=m, num_runs=num_runs, metric=metric))


    #yticks = {'weight': [0, 0.02, 0.04, 0.06, 0.08, 0.10], 'accuracy': [91, 92, 93, 94, 95, 96],
    #          'dead_neurons': [0, 10, 20, 30, 40, 50], 'effective_rank': [0, 10, 20, 30, 40, 50]}[metric]
    
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C9', 'C4', 'C1', 'C0', 'C2']*10,
        #yticks=yticks,
        #xticks=[0, 200*m, 400*m, 600*m, 800*m],
        #xticks_labels=['0', '200', '400', '600', '800'],
        xticks=[0, 250*m],
        xticks_labels=['0', '250'],
        m=m,
        fontsize=16,
        labels=labels,
        filename_pref='cbp_snp_accuracy_tmp'
        #labels=['bp', 'adam', 'l2', 'snp', 'cbp'],
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

