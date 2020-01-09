import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from external.dada.flag_holder import FlagHolder

# options
@click.command()
# target
@click.option('-t', '--target_path', type=str, required=True)

def main(**kwargs):
    plot(**kwargs)

def plot(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # load csv file and plot
    df = pd.read_csv(FLAGS.target_path)
    sns.set()
    ax = sns.lineplot(x="coverage", y="emprical_risk", ci="sd", data=df)
    plt.show()    

if __name__ == '__main__':
    main()