import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import click
import glob
import subprocess

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.plot import plot

# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
@click.option('-x', type=str, required=True)
@click.option('-y', type=str, default='')
@click.option('-a', '--plot_all', is_flag=True, default=False, help='plot all in single image')

def main(**kwargs):
    plot_multi(**kwargs)

def plot_multi(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    run_dir  = '../scripts'
    target_path = os.path.join(FLAGS.target_dir, '**/*.csv')
    weight_paths = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))

    for weight_path in weight_paths:
        # skip 'test.csv'
        if os.path.basename(weight_path) == 'test.csv': continue

        log_dir = os.path.join(os.path.dirname(weight_path), 'plot')
        os.makedirs(log_dir, exist_ok=True)

        basename = os.path.basename(weight_path)
        basename, _ = os.path.splitext(basename) 
        log_path = os.path.join(log_dir, basename)+'.png'

        cmd = 'python plot.py \
            -t {target_dir} \
            -x {x} \
            -s \
            -l {log_path}'.format(
                target_dir=weight_path,
                x=FLAGS.x,
                log_path=log_path)

        # add y
        if FLAGS.y != '':
            cmd += ' -y {y}'.format(y=FLAGS.y)

        # add flag command
        if FLAGS.plot_all:
            cmd += ' --plot_all'

        subprocess.run(cmd.split(), cwd=run_dir)

if __name__ == '__main__':
    main()