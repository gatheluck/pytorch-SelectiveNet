import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import subprocess
import click
import uuid

from external.dada.flag_holder import FlagHolder

# options
@click.command()
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
# optimization
@click.option('--num_epochs', type=int, default=300)
# logging
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('--ex_id', type=str, default=uuid.uuid4().hex, help='id of the experiments')

def main(**kwargs):
    train_multi(**kwargs)

def train_multi(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    run_dir  = '../scripts'
    coverages = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

    for coverage in sorted(coverages):
        suffix  = '_coverage_{:0.2f}'.format(coverage) 
        log_dir = os.path.join(FLAGS.log_dir, FLAGS.ex_id)
        os.makedirs(log_dir, exist_ok=True)

        cmd = 'python train.py \
            -d {dataset} \
            --dataroot {dataroot} \
            --num_epochs {num_epochs} \
            --coverage {coverage} \
            -s {suffix} \
            -l {log_dir}'.format(
                dataset=FLAGS.dataset,
                dataroot=FLAGS.dataroot,
                num_epochs=FLAGS.num_epochs,
                coverage=coverage,
                suffix=suffix,
                log_dir=log_dir)

        subprocess.run(cmd.split(), cwd=run_dir)

if __name__ == '__main__':
    main()