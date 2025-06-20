import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.rcParams['font.family'] = 'Linux Libertine O'  

    gen1(True)
    gen2(True)


def gen1(gen):
    x = np.random.normal(0.5, 0.14, 1000000)

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x_max = 1.0
    bin_width = x_max / 25.
    bins = np.arange(0, x_max + bin_width, bin_width)
    weights = np.ones_like(x) / len(x)*100  # Normalize the histogram
    plt.hist(x, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, weights=weights)
    
    plt.xticks([0.0, 0.5, 1.0], fontsize=20)
    plt.yticks([0, 2.5, 5, 7.5, 10], ['0', '2.5', '5', '7.5', '10'], fontsize=20)
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    
    if gen:
        plt.savefig("histogram-example1.pdf", dpi=500, bbox_inches='tight')
    else:
        plt.show()


def gen2(gen):
    x = np.random.exponential(scale=0.01, size=1000000)  # Exponential distribution

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x_max = 0.5
    bin_width = x_max / 25.
    bins = np.arange(0, x_max + bin_width, bin_width)
    weights = np.ones_like(x) / len(x)*100  # Normalize the histogram
    plt.hist(x, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, weights=weights)
    # plt.hist(x, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xticks([0.0, 0.25, 0.5], ['0', '0.25', '0.5'], fontsize=20)
    plt.yticks([0, 25, 50, 75], fontsize=20)
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    
    if gen:
        plt.savefig("histogram-example2.pdf", dpi=500, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()