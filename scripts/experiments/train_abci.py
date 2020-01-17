import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import subprocess
import click
import uuid

from external.dada.flag_holder import FlagHolder
from external.abci_util.script_generator import generate_script

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
# selective loss
@click.option('--coverage', type=float, default=None)
# at
@click.option('--nb_its', type=int, default=20, help='number of iterations. 20 is the same as Madry et. al., 2018.')
# option for abci
@click.option('--script_root', type=str, required=True)
@click.option('--run_dir', type=str, required=True)
@click.option('--abci_log_dir', type=str, default='~/abci_log')
@click.option('--user', type=str, required=True)
@click.option('--env', type=str, required=True)

def main(**kwargs):
    train_multi(**kwargs)

def train_multi(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # create script output dir
    script_dir = os.path.join(FLAGS.script_root, FLAGS.ex_id)
    os.makedirs(script_dir, exist_ok=True)

    coverages = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70] if not FLAGS.coverage else [FLAGS.coverage]
    ats = ['pgd']
    at_norms = ['linf', 'l2']

    EPS = {
        'pgd-linf': [0, 1, 2, 4, 8, 16],
        'pgd-l2':   [0, 40, 80, 160, 320, 640],
    }

    for coverage in sorted(coverages):
        for at in ats:
            for at_norm in at_norms:
                key = at+'-'+at_norm
                for at_eps in EPS[key]:

                    suffix = '_coverage-{coverage:0.2f}_{at}-{at_norm}_eps-{at_eps:d}'.format(
                        coverage=coverage, at=at, at_norm=at_norm, at_eps=at_eps) 

                    log_dir = os.path.join(FLAGS.log_dir, FLAGS.ex_id)
                    os.makedirs(log_dir, exist_ok=True)

                    cmd = 'python train.py \
                          -d {dataset} \
                          --dataroot {dataroot} \
                          --num_epochs {num_epochs} \
                          --coverage {coverage} \
                          --at {at} \
                          --nb_its {nb_its} \
                          --at_eps {at_eps} \
                          --at_norm {at_norm} \
                          -s {suffix} \
                          -l {log_dir}'.format(
                            dataset=FLAGS.dataset,
                            dataroot=FLAGS.dataroot,
                            num_epochs=FLAGS.num_epochs,
                            coverage=coverage,
                            at=at,
                            nb_its=FLAGS.nb_its,
                            at_eps=at_eps,
                            at_norm=at_norm,
                            suffix=suffix,
                            log_dir=log_dir)

                    script_basename = suffix.lstrip('_')+'.sh'
                    script_path = os.path.join(script_dir, script_basename)
                    generate_script(cmd, script_path, FLAGS.run_dir, FLAGS.abci_log_dir, FLAGS.ex_id, FLAGS.user, FLAGS.env)

if __name__ == '__main__':
    main()