import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from external.dada.logger import Logger
from external.dada.misc import get_time_stamp

def test_logger():
    log_path_root = '/home/gatheluck/Scratch/selectivenet/logs'
    log_basename = 'log_test_'+get_time_stamp('short')
    log_path = os.path.join(log_path_root, log_basename)

    logger = Logger(log_path)

    log_dict  = {'loss01':1.0, 'loss02':2.0}
    log_dict_ = {'loss01':1.0, 'loss03':3.0}
    logger.log(log_dict, 1)
    logger.log(log_dict, 2)
    logger.log(log_dict, 3)
    logger.log(log_dict_, 4)


if __name__ == "__main__":
    test_logger()