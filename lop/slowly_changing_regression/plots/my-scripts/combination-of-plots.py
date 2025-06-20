import sys
import json
import pickle
import os
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot_for_subplot
import matplotlib.gridspec as gridspec


def add_cfg_performance(parent_dir, cfg='', setting_idx=0, m=2*10*1000, num_runs=30, metric='unknown'):
    if metric == 'unknown':
        print('Please specify the metric to be used for performance calculation.')
        quit(0)

    with open(cfg, 'r') as f:
        params = json.load(f)
    # list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = parent_dir + '/' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if metric == 'error':
            # Online performance
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))
        elif metric == 'dead_neurons':
            # Dead neurons
            neurons_max = 5
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'] / neurons_max, m=m)))
        elif metric == 'weights':
            # Weights
            num_weights = 105
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight'] / num_weights, m=m)))
        else:
            print(f'Unknown metric: {metric}')
            quit(0)

    # print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def generate_plot_for_metric(ax, parent_dir, num_runs, metric, m_c, yticks, caption):
    print(f'Plotting metric: {metric}')
    # add plot for all algorithms

    m = {'error': 10000*m_c, 'dead_neurons': 10000*m_c, 'weights': 10000*m_c}[metric]
    
    labels = []
    performances = []
    colors = []

    for algo in all_algos():
        print(f'Adding performance for {algo} with metric {metric} and m={m}')
        labels.append(get_label(algo))
        colors.append(get_color(algo))

        cfg_file = get_cfg_dir(parent_dir, algo)
        performances.append(add_cfg_performance(
            parent_dir=parent_dir, cfg=cfg_file, m=m, num_runs=num_runs, metric=metric
        )) 

    performances = np.array(performances)


    generate_online_performance_plot_for_subplot(
        ax,
        performances=performances,
        colors=colors,
        yticks=yticks,
        # xticks=[0, 500000, 1000000],
        xticks=[0, int(0.5 * 3e6), int(3e6)],
        xticks_labels=['0', '1.5M', '3M'],
        m=m,
        labels=labels,
        fontsize=29,
        caption=caption,
    )



def main():
    plt.rcParams['font.family'] = 'Linux Libertine O'  

    # Combination1: Slowly regression: errors, weight magnitude, dead unit ratio, utility score average
    # combination1()

    # Combination2: histograms for last iteration in slowly regression
    # combination2()

    # Combination3: util maxes for BP and CBP
    # combination3()

    # Appendix: change of histogram, last task

    # one_task = 10*1000
    # total = int(3e6)
    # tasks = total // one_task
    # iterations_to_save = [total-one_task+100, total-one_task+1000, total-one_task+3000, total-1]

    # iterations_to_save = [i // 100 for i in iterations_to_save]
    # iterations_to_save = sort_and_remove_duplicates(iterations_to_save)

    # # # load_util_data_to_file1(iterations_to_save)
    # appendix_util_histograms(iterations_to_save)

    # Appendix second plot: util maxes
    num_runs=100
    # load_appendix_util_maxes(num_runs)
    appendix_util_maxes(num_runs)


def appendix_util_maxes(num_runs):
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    num_runs = 100
    normalize = False
    m = 500

    xticks=[0, int(0.5 * 3e6), int(3e6)]
    xticks_labels=['0', '1.5M', '3M']

    fontsize=29

    iterations_to_save=list(range(int(3e6)))


    with open(f'hist_dump/util_max_all_dump', 'rb+') as f:
        all_performances = pickle.load(f)


    fig, ax = plt.subplots(2, 3, figsize=(22, 6*2))
    labels = ['1st maximum', '2nd maximum', '3rd maximum', '4th maximum', '5th maximum']
    
    generate_online_performance_plot_for_subplot(
        ax[0, 0],
        performances=all_performances[0],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='Backpropagation',
    )

    generate_online_performance_plot_for_subplot(
        ax[0, 1],
        performances=all_performances[1],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='L2 regularization',
    )

    generate_online_performance_plot_for_subplot(
        ax[0, 2],
        performances=all_performances[2],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='Shrink and Perturb',
    )

    generate_online_performance_plot_for_subplot(
        ax[1, 0],
        performances=all_performances[3],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='Continual Backpropagation',
    )

    generate_online_performance_plot_for_subplot(
        ax[1, 1],
        performances=all_performances[4],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='CBP+L2',
    )

    generate_online_performance_plot_for_subplot(
        ax[1, 2],
        performances=all_performances[5],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=fontsize,
        caption='CBP+SnP',
    )

    for r in range(2):
        for c in range(3):
            if r == 0:
                ax[r,c].set_xticklabels([])

            if r == 0:
                ax[r,c].set_yticks([0, 1, 2])
                ax[r,c].set_ylim(-0.03, 2.7)
            elif r == 1:
                ax[r,c].set_yticks([0, 0.5, 1])
                ax[r,c].set_ylim(-0.012, 1.1)

            if c > 0:
                ax[r,c].set_yticklabels([])
            

    # for i in [0, 1]:
    #     ax[i].set_yticks([0, 0.5, 1, 1.5, 2.0])
    #     ax[i].set_ylim(0, 2.1)
    # ax[1].set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(right=0.78, wspace=0.07, hspace=0.2) # for legend

    for a in ax.flatten():
        a.yaxis.grid(False)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=fontsize+2)

    # plt.show()
    plt.savefig('appendix-util-max.pdf', bbox_inches='tight', dpi=500)

    


def appendix_util_histograms(iterations_to_save):
    normalize = False
    num_runs = 100

    y_max = {'bp': 3.9, 'l2': 3.9, 'snp': 2.2, 'cbp': 2.2, 'cbp_l2': 2.2, 'cbp_snp': 2.2}
    # y_labels = {'bp'}
    labels = ['100th iteration', '1000th iteration', '3000th iteration', '10000th iteration']

    algos = ['bp', 'snp', 'cbp']

    fig = plt.figure(figsize=(12*2+2, 18*2*len(algos)//6))  # wider to fit row labels
    gs = gridspec.GridSpec(len(algos), 5, width_ratios=[0.5, 1, 1, 1, 1], wspace=0.1, hspace=0.1)

    # fig, ax = plt.subplots(6, 4, figsize=(12*2, 18*2))

    for (r, algo) in enumerate( algos ):

        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, get_label(algo), va='center', ha='center', fontsize=31, rotation=0)


        with open(f'hist_dump/{algo}_utils_dump', 'rb+') as f:
            util_data_all = pickle.load(f) 

        for (c, _) in enumerate( iterations_to_save ):

            ax = fig.add_subplot(gs[r, c + 1])
            label = labels[c] if r == 0 else ''
            gen_util_plot_for_subplot_given_data(ax, util_data_all[c], num_runs, normalize, 3.2, y_max[algo], label)

            yticks = [i for i in range(0, 10) if i < y_max[algo]]
            ax.set_yticks(yticks)

            if c != 0:
                ax.set_yticklabels([])
            ax.set_xticks([0, 1, 2, 3])
            if r != len(algos)-1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['0', '1', '2', '3'])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


        #ax[0, 0], parent_dir, algo='bp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
         #                     normalize=normalize, x_max=1.0, y_max=230.0, title='Backpropagation')
    
    
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.2, wspace=0.1) 

    # plt.show()
    plt.savefig('appendix-util-hist.pdf', bbox_inches='tight', dpi=500)


def load_util_data_to_file1(iterations_to_save):
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    num_runs=100
    for algo in all_algos():
        util_data = get_cfg_util_data(parent_dir, get_cfg_dir(parent_dir, algo), iterations_to_save, 0, num_runs)
        new_util_data = []
        for (_, iteration_id) in enumerate( iterations_to_save ):
            chosen_util_data = []
            for run_id in range(num_runs):
                chosen_util_data.append(util_data[run_id][iteration_id])
            new_util_data.append(chosen_util_data)
        with open(f'hist_dump/{algo}_utils_dump', 'wb+') as f:
            pickle.dump(new_util_data, f)


def load_appendix_util_maxes(num_runs):
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    normalize = False
    m = 500

    xticks=[0, int(0.5 * 3e6), int(3e6)]
    xticks_labels=['0', '1.5M', '3M']

    iterations_to_save=list(range(int(3e6)))


    all_performances = []

    for algo in all_algos():
        all_performances.append(
            generate_util_maxes_plot_for_algo_for_subplot_get_performances(None, parent_dir, algo, num_runs, normalize, m, 
                                                      iterations_to_save, xticks, xticks_labels, 'Backpropagation')
        )
    
    with open(f'hist_dump/util_max_all_dump', 'wb+') as f:
        pickle.dump(all_performances, f)



def combination3():
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    num_runs = 100
    normalize = False
    m = 500

    xticks=[0, int(0.5 * 3e6), int(3e6)]
    xticks_labels=['0', '1.5M', '3M']
    labels = ['1st maximum', '2nd maximum', '3rd maximum', '4th maximum', '5th maximum']

    iterations_to_save=list(range(int(3e6)))


    with open(f'hist_dump/util_max_all_dump', 'rb+') as f:
        all_performances = pickle.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    generate_online_performance_plot_for_subplot(
        ax[0],
        performances=all_performances[0],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=29,
        caption='Backpropagation',
    )
    generate_online_performance_plot_for_subplot(
        ax[1],
        performances=all_performances[3],
        colors=['C3', 'C4', 'C5', 'C8', 'C9', 'C0'],
        xticks=xticks,
        xticks_labels=xticks_labels,
        m=m * 100, # equal to the util_save_every_nth_iteration
        labels=labels,
        fontsize=29,
        caption='Continual Backpropagation',
    )

    # generate_util_maxes_plot_for_algo_for_subplot(ax[0], parent_dir, 'bp', num_runs, normalize, m, 
    #                                               iterations_to_save, xticks, xticks_labels, 'Backpropagation')
    # generate_util_maxes_plot_for_algo_for_subplot(ax[1], parent_dir, 'cbp', num_runs, normalize, m, 
    #                                               iterations_to_save, xticks, xticks_labels, 'Continual Backpropagation')
    

    for i in [0, 1]:
        ax[i].set_yticks([0, 0.5, 1, 1.5, 2.0])
        ax[i].set_ylim(0, 2.1)
    ax[1].set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(right=0.75, wspace=0.05) # for legend

    for a in ax:
        a.yaxis.grid(False)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30)

    # plt.show()
    plt.savefig('combination-3.pdf', bbox_inches='tight', dpi=500)

    


def combination2():
    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"
    num_runs = 100
    normalize = False

    one_task = 10*1000
    total = int(3e6)
    tasks = total // one_task

    iterations_to_save = [total - 1]  # very last one    

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    gen_util_plot_for_subplot(ax[0, 0], parent_dir=parent_dir, algo='bp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=3.9, title='Backpropagation')
    gen_util_plot_for_subplot(ax[0, 1], parent_dir=parent_dir, algo='l2', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=3.9, title='L2 regularization')
    gen_util_plot_for_subplot(ax[0, 2], parent_dir=parent_dir, algo='snp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=3.9, title='Shrink and Perturb')
    
    gen_util_plot_for_subplot(ax[1, 0], parent_dir=parent_dir, algo='cbp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=1.4, title='CBP')
    gen_util_plot_for_subplot(ax[1, 1], parent_dir=parent_dir, algo='cbp_l2', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=1.4, title='CBP+L2')
    gen_util_plot_for_subplot(ax[1, 2], parent_dir=parent_dir, algo='cbp_snp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=3.5, y_max=1.4, title='CBP+SnP')
    
    for i in [0, 1, 2]:
        ax[1, i].set_yticks([0, 0.5, 1])
        ax[1, i].set_yticklabels(['0', '0.5', '1'])

    for i in [0, 1, 2]:
        ax[0, i].set_xticks([0, 1, 2 ,3])
        ax[1, i].set_xticks([0, 1, 2, 3])

        ax[0, i].set_xticklabels([])
    for (r, c) in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        ax[r, c].set_yticklabels([])

    
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        
    # plt.tight_layout(h_pad=2.0)
    plt.subplots_adjust(hspace=0.2, wspace=0.1) 

    # plt.show()
    plt.savefig('combination-2.pdf', bbox_inches='tight', dpi=500)


def combination1():

    parent_dir = "/home/aldas/TUDelft/RP/results_copied/06-05_slowly-100runs"

    num_runs = 100 # should be 100
    num_runs_util = 100
    m_c = 5
    m_util = 100 * m_c
    
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    # fig, ax = plt.subplots(2, 2)

    generate_plot_for_metric(ax[0, 0], parent_dir=parent_dir, num_runs=num_runs, metric='error', m_c=m_c, yticks=[], caption='Mean squared error')
    generate_plot_for_metric(ax[0, 1], parent_dir=parent_dir, num_runs=num_runs, metric='weights', m_c=m_c, yticks=[], caption='Weight magnitude average')
    generate_plot_for_metric(ax[1, 0], parent_dir=parent_dir, num_runs=num_runs, metric='dead_neurons', m_c=m_c, yticks=[], caption='Ratio of dead neurons')

    generate_mean_plots_for_all_algos_for_subplot(ax[1, 1], parent_dir, num_runs_util, list(range(int(3e6))), False, m_util, 
                                                  [0, int(0.5 * 3e6), int(3e6)], ['0', '1.5M', '3M'])

    plt.tight_layout(h_pad=4.0)
    plt.subplots_adjust(hspace=0.35, right=0.8, wspace=0.2) # for legend

    # prev_handles, prev_labels = ax[0, 0].get_legend_handles_labels()
    # print(prev_handles)
    # print(prev_labels)
    # new_handles = [prev_handles[i] for i in range(0, 11, 2)]

    for a in ax.flatten():
        a.yaxis.grid(False)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30)

    # plt.show()
    plt.savefig('combination-1.pdf', bbox_inches='tight', dpi=500)




if __name__ == '__main__':
    sys.exit(main())

