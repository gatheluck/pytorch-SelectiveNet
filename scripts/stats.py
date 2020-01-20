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
@click.option('-t', '--target_path', type=str, required=True, help='path to test*.csv')
# at
@click.option('--coverage', type=float, default=None)
@click.option('--at', type=str, default=None)
@click.option('--at_norm', type=str, default=None)
@click.option('--at_eps', type=float, default=0.0)
# attack
@click.option('--attack', type=str, default=None)
@click.option('--attack_norm', type=str, default=None)
@click.option('--attack_eps', type=float, default=0.0)
@click.option('--attack_trg_loss', type=str, default=None)


def main(**kwargs):
    stats(**kwargs)

def stats(**kwargs):
    """
    compute statistics of spesific model and spesific attack
    """
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    #FLAGS.summary()

    # load csv file and plot
    df = pd.read_csv(FLAGS.target_path)

    # conditioning data frame
    df = df[df['at']==FLAGS.at] # pgd
    df = df[df['at_norm']==FLAGS.at_norm] # linf /l2
    df = df[df['attack']==FLAGS.attack] # pgd
    df = df[df['attack_norm']==FLAGS.attack_norm] # linf /l2
    df = df[df['attack_trg_loss']==FLAGS.attack_trg_loss] # both / cls / rjc
    df = df[df['at_eps']==FLAGS.at_eps]
    df = df[df['attack_eps']==FLAGS.attack_eps]
    df = df[df['coverage']==FLAGS.coverage]

    df_dict = df[["accuracy", "rejection rate", "rejection precision"]].describe().to_dict()
    
    return df_dict
    
if __name__ == '__main__':
    main()