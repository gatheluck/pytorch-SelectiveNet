import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import click
import uuid
import glob

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.test import test

# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')

def main(**kwargs):
    test_multi(**kwargs)

def test_multi(**kwargs):
    """
    this script loads all 'weight_final_{something}.pth' files which exisits under 'kwargs.target_dir' and execute test.
    if there is exactly same file, the result becomes the mean of them.
    the results are saved as csv file.

    'target_dir' should be like follow

    ~/target_dir/XXXX/weight_final_coverage_0.10.pth
                     /weight_final_coverage_0.95.pth
                     /weight_final_coverage_0.90.pth
                     ...
                /YYYY/weight_final_coverage_0.10.pth
                     /weight_final_coverage_0.95.pth
                     /weight_final_coverage_0.90.pth
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

    for weight_path in weight_paths:
        # get coverage
        # name should be like, '~_coverage_{}.pth'
        basename = os.path.basename(weight_path)
        basename, ext = os.path.splitext(basename)
        coverage = float(basename.split('_')[-1])

        # keyword args for test function
        # variable args
        kw_args = {}
        kw_args['weight'] = weight_path
        kw_args['dataset'] = FLAGS.dataset
        kw_args['dataroot'] = FLAGS.dataroot
        kw_args['coverage'] = coverage
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
        metric_dict['coverage'] = coverage
        metric_dict['path'] = weight_path
        metric_dict.update(out_dict)

        # log
        logger.log(metric_dict)

if __name__ == '__main__':
    main()