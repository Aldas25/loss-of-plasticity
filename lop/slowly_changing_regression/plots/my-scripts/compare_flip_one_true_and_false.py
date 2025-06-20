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

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)

def generate_plot(ax, parent_dir, cfg_file, m, num_runs, title):
    performances = []

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    _, param_settings = get_configurations(params=params)
    
    labels = [p[0] for p in param_settings]
    for i in range(len(param_settings)):
        performances.append(add_cfg_performance(parent_dir=parent_dir, cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))
    performances = np.array(performances)

    generate_online_performance_plot_for_subplot(
        ax,
        performances=performances,
        colors=['C0', 'C1', 'C2'],
        # yticks=yticks,
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,
        labels=labels,
        fontsize=29,
        caption=title,
    )




def main(arguments):
    plt.rcParams['font.family'] = 'Linux Libertine O'  

    parent_dir_true = "/home/aldas/TUDelft/RP/results_copied/05-22_result-backup/slowly_flip-one-true_5runs"
    parent_dir_false = "/home/aldas/TUDelft/RP/results_copied/05-22_result-backup/slowly_flip-one-false_5runs"
    
    m = 5 * 10000 * 2
    num_runs = 5

    cfg_suf = "/cfg/sgd/bp/relu.json" 

    cfg_file_true = parent_dir_true + cfg_suf
    cfg_file_false = parent_dir_false + cfg_suf

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    generate_plot(ax[0], parent_dir_true, cfg_file_true, m, num_runs, 'Flipping one bit')
    generate_plot(ax[1], parent_dir_false, cfg_file_false, m, num_runs, 'Flipping all bits')

    # plt.tight_layout(h_pad=4.0)
    plt.subplots_adjust(right=0.8, wspace=0.05) # for legend

    for a in ax:
        a.yaxis.grid(False)

        a.set_ylim(0.4, 1.75)
        a.set_yticks([0.5, 1, 1.5])

    ax[1].set_yticklabels([])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30)
    plt.savefig('comparison-flip-one.pdf', bbox_inches='tight', dpi=500)





if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

