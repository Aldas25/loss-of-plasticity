import sys
import json
import pickle
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

def generate_plot_for_metric(parent_dir, num_runs, metric, algos, pref, m_c):
    mult_const = {'weight': 10*1000, 'accuracy': 10*1000, 'dead_neurons': 1}[metric]
    m = {'weight': mult_const*m_c, 'accuracy': mult_const*m_c, 'dead_neurons': mult_const*m_c}[metric]

    performances = []

    labels = []
    colors = []

    for algo in algos:
        labels.append(get_label(algo))
        colors.append(get_color(algo))

        performances.append(add_cfg_performance(
            parent_dir=parent_dir, cfg=get_cfg_dir(parent_dir, algo), m=m, num_runs=num_runs, metric=metric))


    generate_online_performance_plot(
        performances=performances,
        colors=colors,
        xticks=[0, 400*mult_const, 800*mult_const],
        xticks_labels=['0', '400', '800'],
        m=m,
        fontsize=16,
        labels=labels,
        filename_pref=f'{pref}_{metric}_runs={num_runs}',
        svg=True,
    )




def main():
    num_runs = 30  # should be 30
    parent_dir = '/home/aldas/TUDelft/RP/results_copied/06-07_permuted-30runs'

    m_c = 15

    for metric in ['weight', 'accuracy', 'dead_neurons']:
        print(f'-'*20)
        print(f'Plotting metric: {metric}')

        # generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric, algos=all_algos(), pref='mnist_all', m_c=m_c)
        # generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric, algos=all_algos()[1:], pref='mnist_without_bp', m_c=m_c)

        generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric, algos=['cbp_snp'], pref='cbp_snp', m_c=m_c)
        # generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric, algos=['cbp', 'cbp_l2', 'cbp_snp'], pref='cbp_variations')
        generate_plot_for_metric(parent_dir=parent_dir, num_runs=num_runs, metric=metric, algos=['cbp', 'cbp_snp'], pref='cbp_variations', m_c=m_c)

    

if __name__ == '__main__':
    sys.exit(main())

