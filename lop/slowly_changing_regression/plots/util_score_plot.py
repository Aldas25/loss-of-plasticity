import sys
import json
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot

def create_histogram(filename='util_histogram.png', title='title', util_data=None, normalize=False):
    data = np.array([t.numpy() for t in util_data])
    print(f'{title}, data: {data}')

    if normalize: # TODO
        print(f'TODO NORMALIZATION ')
    


    fig, ax = plt.subplots(figsize=(10, 6))
    num_bins = 20
    hist, bins, patches = ax.hist(data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    # ax.axvline(np.mean(util_data), color='red', linestyle='dashed', linewidth=2, label='Mean')

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # change the cfg file to get the results for different activation functions, ex. '../cfg/sgd/bp/tanh.json'
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/sgd/bp/relu.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    performances = []
    m = int(params['flip_after'])*2

    _, param_settings = get_configurations(params=params)
    labels = param_settings
    num_runs = params['num_runs']
    indices = [i for i in range(len(param_settings))]
    for setting_idx in indices:
        per_param_setting_performance = []
        for idx in range(num_runs):
            util_save_dir = params['data_dir'].replace("data", "utils_saved")
            util_save_dir = os.path.join(os.path.pardir, util_save_dir, str(setting_idx), str(idx))
            util_save_file = os.path.join(util_save_dir, 'util')
            bias_corrected_util_save_file = os.path.join(util_save_dir, 'bias_corrected_util')
            with open(util_save_file, 'rb') as f:
                util_data = pickle.load(f)

            with open(bias_corrected_util_save_file, 'rb') as f:
                bias_corrected_util_data = pickle.load(f)

            print(f'setting_idx: {setting_idx}, idx: {idx}, data size: {len(util_data)}, {len(bias_corrected_util_data)}')

            create_histogram(filename=os.path.join(util_save_dir, 'util_histogram.png'), 
                             title='util_histogram', 
                             util_data=util_data[300], normalize=False)
            create_histogram(filename=os.path.join(util_save_dir, 'util_histogram_normalized.png'),
                             title='util_histogram_normalized', 
                             util_data=util_data[300], normalize=True)
            create_histogram(filename=os.path.join(util_save_dir, 'bias_corrected_util_histogram.png'), 
                             title='bias_corrected_util_histogram',
                             util_data=util_data[300], normalize=False)
            create_histogram(filename=os.path.join(util_save_dir, 'bias_corrected_util_histogram_normalized.png'), 
                             title='bias_corrected_util_histogram_normalized',
                             util_data=util_data[300], normalize=True)

            # Online performance
            # per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))

        # print(param_settings[setting_idx], setting_idx)
        # return np.array(per_param_setting_performance)

    #     performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))
    # # performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/bp/linear.json', setting_idx=0, m=m, num_runs=num_runs))
    # # labels.append('linear')
    # performances = np.array(performances)

    # if params['hidden_activation'] in ['relu', 'swish', 'leaky_relu']:
    #     yticks = [0.6, 0.8, 1., 1.2, 1.4]
    # else:
    #     yticks = [0.4, 0.6, 0.8, 1, 1.2]
    # print(yticks, params['hidden_activation'])
    # generate_online_performance_plot(
    #     performances=performances,
    #     colors=['C3', 'C4', 'C5', 'C8'],
    #     yticks=yticks,
    #     xticks=[0, 500000, 1000000],
    #     xticks_labels=['0', '0.5M', '1M'],
    #     m=m,
    #     labels=labels
    # )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

