import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import re
import click

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.stats import stats

# options
@click.command()
@click.option('-t', '--target_path', type=str, required=True, help='path to test*.csv')
# at
@click.option('--coverage', type=float, default=None)
# attack
@click.option('--attack', type=str, default='pgd')
@click.option('--attack_norm', type=str, required=True)
@click.option('--attack_trg_loss', type=str, required=True)

def main(**kwargs):
    stats_multi(**kwargs)

def stats_multi(**kwargs):
    # flags
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    at = 'pgd'
    at_norms = ['linf', 'l2']
    at_epses = {'linf':[0, 4, 8],
                'l2'  :[0, 40, 80, 160]}
    
    attack = FLAGS.attack
    attack_norm = FLAGS.attack_norm
    attack_epses = {'linf':[0, 4, 8, 16],
                    'l2'  :[0, 80, 160]}

    for at_norm in at_norms:
        for at_eps in at_epses[at_norm]:
            for attack_eps in attack_epses[attack_norm]:

                kw_args = {}
                kw_args['target_path'] = FLAGS.target_path
                kw_args['coverage'] = FLAGS.coverage
                kw_args['at'] = at
                kw_args['attack'] = attack
                kw_args['at_norm'] = at_norm
                kw_args['attack_norm'] = attack_norm
                kw_args['at_eps'] = at_eps
                kw_args['attack_eps'] = attack_eps
                kw_args['attack_trg_loss'] = FLAGS.attack_trg_loss

                df_dict = stats(**kw_args)

                template = 'AT:{at}-{at_norm:<5s}: {at_eps:>3d}　|　Att: {attack}-{attack_norm:<5s}: {attack_eps:>3d} | Acc: {acc_mean:0.1f}+-{acc_std:0.2f} | Rej: {rjc_mean:0.1f}+-{rjc_std:0.2f} | PR:  {pr_mean:0.1f} +-{pr_std:0.2f}'.format(
                    at=at, at_norm=at_norm, at_eps=at_eps, \
                    attack=attack, attack_norm=attack_norm, attack_eps=attack_eps, \
                    acc_mean=df_dict['accuracy']['mean']*100, acc_std=df_dict['accuracy']['std']*100, \
                    rjc_mean=df_dict['rejection rate']['mean']*100, rjc_std=df_dict['rejection rate']['std']*100, \
                    pr_mean=df_dict['rejection precision']['mean']*100, pr_std=df_dict['rejection precision']['std']*100)

                print(template)

if __name__ == '__main__':
    main()