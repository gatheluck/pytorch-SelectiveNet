import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_model
from external.dada.io import load_model
from external.dada.logger import Logger
from external.advex_uar.advex_uar.attacks.pgd_attack import PGDAttackVariant
from external.advex_uar.advex_uar.common.pyt_common import get_step_size

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=512)
@click.option('--dropout_prob', type=float, default=0.3)
@click.option('-w', '--weight', type=str, required=True, help='model weight path')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='/home/gatheluck/Scratch/selectivenet/data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=128)
@click.option('--normalize', is_flag=True, default=True)
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
# adversarial attack
@click.option('--attack', type=str, default=None)
@click.option('--nb_its', type=int, default=10)
@click.option('--eps_max', type=float, default=0.0)
@click.option('--step_size', type=float, default=None)
@click.option('--norm', type=str, default='linf')
@click.option('--mode', type=str, default='both')


def main(**kwargs):
    test(**kwargs)

def test(**kwargs):
    """
    test 
    """
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    test_dataset   = dataset_builder(train=False, normalize=FLAGS.normalize)
    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    model = SelectiveNet(features, FLAGS.dim_features, dataset_builder.num_classes).cuda()
    load_model(model, FLAGS.weight)

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # loss
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_selective = SelectiveLoss(base_loss, coverage=FLAGS.coverage)

    # adversarial attack
    if FLAGS.attack:
        # get step_size
        if not FLAGS.step_size:
            FLAGS.step_size = get_step_size(FLAGS.eps_max, FLAGS.nb_its)

        # create attacker
        if FLAGS.attack=='pgd':
            attacker = PGDAttackVariant(
                        FLAGS.nb_its, FLAGS.eps_max, FLAGS.step_size, dataset=FLAGS.dataset, 
                        coverage=FLAGS.coverage, norm=FLAGS.norm, mode=FLAGS.mode)
        else:
            raise NotImplementedError('invalid attack method.')
    
    # pre epoch
    test_metric_dict = MetricDict()

    # test
    for i, (x,t) in enumerate(test_loader):
        model.eval()
        x = x.to('cuda', non_blocking=True)
        t = t.to('cuda', non_blocking=True)
        loss_dict = OrderedDict()

        # adversarial samples
        if FLAGS.attack and FLAGS.eps_max>0:
            # create adversarial sampels
            model.zero_grad()
            x = attacker(model, x.detach(), t.detach())

        with torch.autograd.no_grad():
            model.zero_grad()
            # forward
            out_class, out_select, out_aux = model(x)
            
            # compute selective loss
            selective_loss, loss_dict = criterion_selective(out_class, out_select, t)
            selective_loss *= FLAGS.alpha
            loss_dict['selective_loss'] = selective_loss.detach().cpu().item()

            # compute standard cross entropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(out_aux, t)
            ce_loss *= (1.0 - FLAGS.alpha)
            loss_dict['ce_loss'] = ce_loss.detach().cpu().item()

            # total loss
            loss = selective_loss + ce_loss
            loss_dict['loss'] = loss.detach().cpu().item()

            # evaluation
            evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())
            loss_dict.update(evaluator())

        test_metric_dict.update(loss_dict)

    # post epoch
    print_metric_dict(None, None, test_metric_dict.avg, mode='test')

    return test_metric_dict.avg

if __name__ == '__main__':
    main()