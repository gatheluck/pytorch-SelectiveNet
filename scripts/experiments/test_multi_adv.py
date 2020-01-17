import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import re
import click
import uuid
import glob

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.test import test

EPS = {
    'pgd_linf': [0, 1, 2, 4, 8, 16],
    'pgd_l2':   [0, 40, 80, 160, 320, 640],
}


# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
# adversarial attack
@click.option('--attack', type=str, required=True)
@click.option('--attack_norm', type=str, required=True)
@click.option('--nb_its', type=int, default=10)
@click.option('--attack_trg_loss', type=str, required=True)

def main(**kwargs):
    test_multi(**kwargs)

def parse_weight_basename(weight_basename):
    ret_dict = dict()

    # remove ext
    basename, ext = os.path.splitext(weight_basename)

    # extract coverage and else
    # \d : any single number
    # \w : any single number or charactor
    # .  : any single charactor
    # +  : sequence more than one time
    # *  : sequence more than zero time

    # 'weight_final_coverage_{coverage}_{else}'
    pattern = r'weight_final_coverage_(\d.\d+)_(.*)'
    result = re.match(pattern, basename)

    ret_dict['coverage'] = float(result.group(1))

    at_info = result.group(2)

    if at_info == 'std':
        ret_dict['at'] = None
        ret_dict['at_norm'] = None
        ret_dict['at_eps'] = 0.0
    else:
        # '{at}-{at_norm}_eps-{at_eps}' 
        pattern = r'(\w+)-(\w+)_eps-(\d+)'
        result = re.match(pattern, at_info)

        ret_dict['at'] = result.group(1)
        ret_dict['at_norm'] = result.group(2)
        ret_dict['at_eps'] = float(result.group(3))
        
    return ret_dict

def test_multi(**kwargs):
    """
    this script loads all 'weight_final_{something}.pth' files which exisits under 'kwargs.target_dir' and execute test.
    if there is exactly same file, the result becomes the mean of them.
    the results are saved as csv file.

    'target_dir' should be like follow
    (.pth file name should be "weight_final_coverage_{}") 

    ~/target_dir/XXXX/weight_final_coverage_0.10_pgd-linf_eps-0.pth
                     ...
                     /weight_final_coverage_0.10_pgd-linf_eps-8.pth
                     /weight_final_coverage_0.10_pgd-linf_eps-16.pth
                     ...
                /YYYY/weight_final_coverage_0.10_pgd-linf_eps-0.pth
                     ...
                     /weight_final_coverage_0.10_pgd-linf_eps-8.pth
                     /weight_final_coverage_0.10_pgd-linf_eps-16.pth
                     ...
    """
    # flags
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # paths
    run_dir  = '../scripts'
    target_path = os.path.join(FLAGS.target_dir, '**/weight_final*.pth')
    weight_paths = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))
    log_path = os.path.join(FLAGS.target_dir, 'test.csv')

    # logging
    logger = Logger(path=log_path, mode='test')

    # get epses
    key = FLAGS.attack + '_' + FLAGS.attack_norm
    attack_epses = EPS[key]

    for weight_path in weight_paths:
        for attack_eps in attack_epses:

            # parse basename
            basename = os.path.basename(weight_path)
            ret_dict = parse_weight_basename(basename)

            # keyword args for test function
            # variable args
            kw_args = {}
            kw_args['weight'] = weight_path
            kw_args['dataset'] = FLAGS.dataset
            kw_args['dataroot'] = FLAGS.dataroot
            kw_args['coverage'] = ret_dict['coverage']
            kw_args['attack'] = FLAGS.attack
            kw_args['nb_its'] = FLAGS.nb_its
            kw_args['step_size'] = None
            kw_args['attack_eps'] = attack_eps
            kw_args['attack_norm'] = FLAGS.attack_norm
            kw_args['attack_trg_loss'] = FLAGS.attack_trg_loss

            # default args
            kw_args['dim_features'] = 512
            kw_args['dropout_prob'] = 0.3
            kw_args['num_workers'] = 8
            kw_args['batch_size'] = 128
            kw_args['normalize'] = True
            kw_args['alpha'] = 0.5
            
            # run test
            out_dict = test(**kw_args)

            metric_dict = OrderedDict()
            metric_dict['coverage'] = ret_dict['coverage']
            # at
            metric_dict['at'] = ret_dict['at']
            metric_dict['at_norm'] = ret_dict['at_norm']
            metric_dict['at_eps'] = ret_dict['at_eps']
            # attack
            metric_dict['attack'] = FLAGS.attack 
            metric_dict['attack_norm'] = FLAGS.attack_norm
            metric_dict['attack_eps'] = attack_eps
            metric_dict['attack_trg_loss'] = FLAGS.attack_trg_loss
            # path
            metric_dict['path'] = weight_path
            metric_dict.update(out_dict)

            # log
            logger.log(metric_dict)

if __name__ == '__main__':
    main()