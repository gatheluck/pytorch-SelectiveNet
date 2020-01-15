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
@click.option('-t', '--target_path', type=str, required=True)
@click.option('-x', type=str, required=True)
@click.option('-y', type=str, default='')
@click.option('--plot_all', is_flag=True, default=False, help='plot all in single image')
@click.option('--plot_test', is_flag=True, default=False, help='plot test.csv file')
@click.option('-l', '--log_path', type=str, default='', help='path of log')
@click.option('-s', '--save', is_flag=True, default=False, help='save results')

def main(**kwargs):
    plot(**kwargs)

def plot(**kwargs):
    """
    reference
    - https://qiita.com/ryo111/items/bf24c8cf508ad90cfe2e (how to make make block)
    - https://heavywatal.github.io/python/matplotlib.html
    """
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    if (FLAGS.plot_all is True) and (FLAGS.plot_test is True):
        raise ValueError('invalid option. either "plot_all" or "plot_test" should be True.')

    # load csv file and plot
    df = pd.read_csv(FLAGS.target_path)

    # plot all variable. this is basically used for visualize training log.
    if FLAGS.plot_all:
        # ignore some columns
        ignore_columns = ['Unnamed: 0', 'time stamp', 'step', FLAGS.x]
        column_names = [column for column in df.columns if column not in ignore_columns]
        
        # create figure
        fig = plt.figure(figsize=(4*len(column_names),3))

        for i, column_name in enumerate(column_names):
            ax = fig.add_subplot(1, len(column_names), i+1)
            sns.lineplot(x=FLAGS.x, y=column_name, ci="sd", data=df)
                
        plt.tight_layout()

    # plot test.csv file. 
    elif FLAGS.plot_test:
        # ignore some columns
        ignore_columns = ['Unnamed: 0', 'time stamp', 'path', 'loss', 'selective_loss', FLAGS.x]
        column_names = [column for column in df.columns if column not in ignore_columns]

        # create figure
        fig = plt.figure(figsize=(4*len(column_names),3))

        for i, column_name in enumerate(column_names):
            ax = fig.add_subplot(1, len(column_names), i+1)
            sns.lineplot(x=FLAGS.x, y=column_name, ci="sd", data=df)
                
        plt.tight_layout()

    # plot specified variable
    else:
        if FLAGS.y == '':
            raise ValueError('please specify "y"')
        fig = plt.figure()
        ax = fig.subplots()
        sns.lineplot(x=FLAGS.x, y=FLAGS.y, ci="sd", data=df)
    
    # show and save
    if FLAGS.save:
        plt.close()
        if FLAGS.log_path == '': 
            raise ValueError('please specify "log_path"')
        os.makedirs(os.path.dirname(FLAGS.log_path), exist_ok=True)
        fig.savefig(FLAGS.log_path)
    else:
        plt.show()
    

if __name__ == '__main__':
    main()