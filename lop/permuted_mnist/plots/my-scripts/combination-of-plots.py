import sys
import json
import pickle
import matplotlib.gridspec as gridspec
import os
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot_for_subplot



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
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m)))

    per_param_np_arr = np.array(per_param_setting_performance)
    print(param_settings[setting_idx], setting_idx, per_param_np_arr.mean())
    return per_param_np_arr

def generate_plot_for_metric(ax, parent_dir, num_runs, metric, algos, m_c, caption):
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


    generate_online_performance_plot_for_subplot(
        ax,
        performances=performances,
        colors=colors,
        xticks=[0, 400*mult_const, 800*mult_const],
        xticks_labels=['0', '400', '800'],
        m=m,
        fontsize=29,
        labels=labels,
        caption=caption,
    )



def main():
    plt.rcParams['font.family'] = 'Linux Libertine O' 

    # Combination1: MNIST: errors, weight magnitude, dead unit ratio, utility score average
    # combination1()

    # Combination2: histograms for last iteration in slowly regression
    # combination2()

    # Combination3: util maxes for BP and CBP
    # combination3()

    # # for appendix, load data to file
    # start_it = 799*10*1000 # beginning of last task
    # iterations_to_save = [start_it+100, start_it+1000, start_it+3000, start_it+10000-1]
    # iterations_to_save = [i // 100 for i in iterations_to_save]
    # iterations_to_save = sort_and_remove_duplicates(iterations_to_save)

    # # load_util_data_to_file1(iterations_to_save)
    # appendix_util_histograms(iterations_to_save)

    # appendix two: util maxes for MNIST
    # num_runs = 30
    # load_appendix_util_maxes(num_runs)
    appendix_util_maxes()


def appendix_util_maxes():
    parent_dir = "..." # fill in the folder path
    num_runs = 30  # should be 30
    num_inputs = 10*1000

    xticks=[0, 400*num_inputs, 800*num_inputs]
    xticks_labels=['0', '400', '800']
    
    m = 500
    normalize = True
    fontsize=29

    num_tasks = 800
    total_iterations = num_tasks * num_inputs

    iterations_to_save=list(range(total_iterations)) # all tasks

    with open(f'hist_dump/util_max_all_dump', 'rb+') as f:
        all_performances = pickle.load(f)


    fig, ax = plt.subplots(2, 3, figsize=(22, 6*2))
    labels = ['1-50', '51-100', '101-150', '151-200', '201-250']

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
                ax[r,c].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
                ax[r,c].set_ylim(-0.01, 0.81)
            elif r == 1:
                ax[r,c].set_yticks([0, 0.2, 0.4, 0.6])
                ax[r,c].set_ylim(-0.008, 0.7)

            if c > 0:
                ax[r,c].set_yticklabels([])
            

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, wspace=0.07, hspace=0.2) # for legend

    for a in ax.flatten():
        a.yaxis.grid(False)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=fontsize+2)

    # plt.show()
    plt.savefig('mnist-appendix-util-max.pdf', bbox_inches='tight', dpi=500)

    


def load_appendix_util_maxes(num_runs):
    parent_dir = "..." # fill in the folder path
    num_inputs = 10*1000

    xticks=[0, 400*num_inputs, 800*num_inputs]
    xticks_labels=['0', '400', '800']
    
    m = 500
    normalize = True

    num_tasks = 800
    total_iterations = num_tasks * num_inputs

    iterations_to_save=list(range(total_iterations)) # all tasks

    all_performances = []

    for algo in all_algos():
        all_performances.append(
            generate_util_maxes_plot_for_algo_for_subplot_get_performances(None, parent_dir, algo, num_runs, normalize, m, 
                                                      iterations_to_save, xticks, xticks_labels, '', labels_mnist=True)
        )
    
    with open(f'hist_dump/util_max_all_dump', 'wb+') as f:
        pickle.dump(all_performances, f)


def appendix_util_histograms(iterations_to_save):
    normalize = True
    num_runs = 30

    y_max = {'bp':230, 'l2': 160, 'snp': 160, 'cbp': 170, 'cbp_l2': 130, 'cbp_snp': 130}
    labels = ['100th iteration', '1000th iteration', '3000th iteration', '10000th iteration']

    algos = ['bp', 'l2']

    fig = plt.figure(figsize=(12*2+2, 18*2*len(algos)//6))  # wider to fit row labels
    gs = gridspec.GridSpec(len(algos), 5, width_ratios=[0.5, 1, 1, 1, 1], wspace=0.1, hspace=0.1)

    # fig, ax = plt.subplots(6, 4, figsize=(12*2, 18*2))

    for (r, algo) in enumerate( algos ):

        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, get_label(algo), va='center', ha='center', fontsize=29, rotation=0)


        with open(f'hist_dump/{algo}_utils_dump', 'rb+') as f:
            util_data_all = pickle.load(f) 

        for (c, _) in enumerate( iterations_to_save ):

            ax = fig.add_subplot(gs[r, c + 1])
            label = labels[c] if r == 0 else ''
            gen_util_plot_for_subplot_given_data(ax, util_data_all[c], num_runs, normalize, 1, y_max[algo], label)

            if c != 0:
                ax.set_yticklabels([])
            ax.set_xticks([0, 0.5, 1])
            if r != len(algos) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(['0', '0.5', '1'])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.savefig('appendix-util-hist.pdf', bbox_inches='tight', dpi=500)


def load_util_data_to_file1(iterations_to_save):
    parent_dir = "..." # fill in the folder path
    num_runs=30
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


def combination3():
    parent_dir = "..." # fill in the folder path
    num_runs = 30  # should be 30
    num_inputs = 10*1000

    xticks=[0, 400*num_inputs, 800*num_inputs]
    xticks_labels=['0', '400', '800']
    
    m = 500
    normalize = True

    num_tasks = 800
    total_iterations = num_tasks * num_inputs

    iterations_to_save=list(range(total_iterations)) # all tasks

    with open(f'hist_dump/util_max_all_dump', 'rb+') as f:
        all_performances = pickle.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    labels = ['1-50', '51-100', '101-150', '151-200', '201-250']
    

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

    for i in [0, 1]:
        ax[i].set_ylim(0, 0.8)
    ax[1].set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(right=0.83, wspace=0.05) # for legend

    for a in ax:
        a.yaxis.grid(False)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30)

    # plt.show()
    plt.savefig('mnist-combination-3.pdf', bbox_inches='tight', dpi=500)

    


def combination2():
    parent_dir = "..." # fill in the folder path

    one_task = 10*1000
    num_tasks = 800
    normalize = True
    total = num_tasks * one_task

    iterations_to_save = [int(total - 1)]  # very last one
    num_runs = 30

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    gen_util_plot_for_subplot(ax[0, 0], parent_dir=parent_dir, algo='bp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=230.0, title='Backpropagation')
    gen_util_plot_for_subplot(ax[0, 1], parent_dir=parent_dir, algo='l2', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=230.0, title='L2 regularization')
    gen_util_plot_for_subplot(ax[0, 2], parent_dir=parent_dir, algo='snp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=230.0, title='Shrink and Perturb')
    
    gen_util_plot_for_subplot(ax[1, 0], parent_dir=parent_dir, algo='cbp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=145, title='CBP')
    gen_util_plot_for_subplot(ax[1, 1], parent_dir=parent_dir, algo='cbp_l2', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=145, title='CBP+L2')
    gen_util_plot_for_subplot(ax[1, 2], parent_dir=parent_dir, algo='cbp_snp', num_runs=num_runs, iterations_to_save=iterations_to_save, 
                              normalize=normalize, x_max=1.0, y_max=145, title='CBP+SnP')
    

    for i in [0, 1, 2]:
        ax[0, i].set_xticklabels([])
    for (r, c) in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        ax[r, c].set_yticklabels([])

    
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        
    plt.subplots_adjust(hspace=0.2, wspace=0.1) 

    plt.savefig('mnist-combination-2.pdf', bbox_inches='tight', dpi=500)


def combination1():

    num_runs = 30  # should be 30
    num_runs_util = 30
    parent_dir = '...' # fill in the folder path

    m_c = 15
    m_util = 100 * m_c

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))

    generate_plot_for_metric(ax[0, 0], parent_dir=parent_dir, num_runs=num_runs, metric='accuracy', algos=all_algos(), 
                             m_c=m_c, caption='Online accuracy')
    generate_plot_for_metric(ax[0, 1], parent_dir=parent_dir, num_runs=num_runs, metric='weight', algos=all_algos(),
                             m_c=m_c, caption='Weight magnitude average')
    generate_plot_for_metric(ax[1, 0], parent_dir=parent_dir, num_runs=num_runs, metric='dead_neurons', algos=all_algos(),
                             m_c=m_c, caption='Ratio of dead neurons')

    generate_mean_plots_for_all_algos_for_subplot(ax[1, 1], parent_dir, num_runs_util, list(range(800 * 10*1000)), True, m_util, 
                                                  [0, 400*10*1000, 800*10*1000], ['0', '400', '800'])

    plt.tight_layout(h_pad=4.0)
    plt.subplots_adjust(hspace=0.35, right=0.79, wspace=0.2) # for legend


    for a in ax.flatten():
        a.yaxis.grid(False)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30)

    # plt.show()
    plt.savefig('mnist-combination-1.pdf', bbox_inches='tight', dpi=500)




if __name__ == '__main__':
    sys.exit(main())

