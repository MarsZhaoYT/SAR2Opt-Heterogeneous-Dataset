import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def main(opt):
    in_dir = opt.in_dir
    df = pd.read_csv(in_dir, header=None)

    ax = df.plot.bar(x=0, y=[1,2,3], legend=True, rot=50, fontsize=12, width=0.7)
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.legend(['Epoch 20', 'Epoch 40', 'Epoch 60'])
    plt.show()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str)
    opt = parser.parse_args()
    main(opt)