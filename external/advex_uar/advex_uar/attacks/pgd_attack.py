import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../')
sys.path.append(base)
sys.path.append(root)

from advex_uar.attacks.attacks import AttackWrapper
from selectivenet.loss import SelectiveLoss


class PGDAttackVariant(AttackWrapper):
    """
    PGD Attack variant for prediction with rejection.

    """
    def __init__(self, nb_its, eps_max, step_size, dataset, coverage, norm='linf', rand_init=True, scale_each=False, mode='both'):
        """
        Parameters:
            nb_its (int):          Number of PGD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            dataset (str):         dataset name
            coverage (float)       coverage of selective loss
            norm (str):            Either 'linf' or 'l2'
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            mode (str):            Attack mode 
        """
        if mode not in ['both', 'rjc', 'cls']:
            raise ValueError 

        super().__init__(dataset)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.dataset = dataset
        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.coverage = coverage
        self.mode = mode
        self.nb_backward_steps = self.nb_its
        
    def _run_one(self, pixel_model, pixel_inp, delta, target, eps, step_size, avoid_target=True):
        # attack rejection model. 
        out_class, out_select, out_aux = pixel_model(pixel_inp + delta)
        
        if self.norm == 'l2':
            l2_max = eps
        

        for it in range(self.nb_its):
            base_loss = torch.nn.CrossEntropyLoss(reduction='none')
            SelectiveCELoss = SelectiveLoss(base_loss, coverage=self.coverage)
            # compute selective loss
            selective_loss, loss_dict = SelectiveCELoss(out_class, out_select, target)
            # compute standard cross entropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(out_aux, target)

            if self.mode == 'rjc':
                loss = selective_loss
            elif self.mode == 'cls':
                loss = ce_loss
            else: # both
                loss = (0.5*selective_loss) + (0.5*ce_loss)

            loss.backward(retain_graph=True)
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = delta.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None, None, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, eps[:, None, None, None]), -eps[:, None, None, None])
                delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None, None, None]                
                delta.data = delta.data + step_size[:, None, None, None] * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None, None, None]
                delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                s = pixel_model(pixel_inp + delta)
                delta.grad.data.zero_()
        return delta

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')
    
    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand.mul(self.eps_max)
            step_size = rand.mul(self.step_size)
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        delta = self._init(pixel_inp.size(), base_eps)
        if self.nb_its > 0:
            delta = self._run_one(pixel_model, pixel_inp, delta, target, base_eps,
                                  step_size, avoid_target=avoid_target)
        else:
            delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
        pixel_result = pixel_inp + delta
        return pixel_result

if __name__ == '__main__':
    from selectivenet.vgg_variant import vgg16_variant
    from selectivenet.model import SelectiveNet
    from selectivenet.loss import SelectiveLoss
    from selectivenet.data import DatasetBuilder
    from external.dada.io import load_model

    # dataset
    dataset_builder = DatasetBuilder(name='cifar10', root_path='../../../../data')
    test_dataset   = dataset_builder(train=False, normalize=True)
    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, 0.3).cuda()
    model = SelectiveNet(features, 512, dataset_builder.num_classes).cuda()

    weight_path = '../../../../logs/abci/cifar10_ex01/0f7d6935b74b42c0a705dd241635d76b/weight_final_coverage_0.70.pth'
    load_model(model, weight_path)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # attacker
    attacker = PGDAttackVariant(10, 32, 30, 'cifar10', coverage=0.7, mode='both')

    # test
    for i, (x,t) in enumerate(test_loader):
        model.eval()
        x = x.to('cuda', non_blocking=True)
        t = t.to('cuda', non_blocking=True)

        x_adv = attacker(model, x, t)

        with torch.autograd.no_grad():
            # forward
            class_std, select_std, aux_std = model(x)
            class_adv, select_adv, aux_adv = model(x_adv)

            print('std: {}/{} corrects'.format((t==torch.argmax(class_std, dim=1)).sum(), t.size(0)))
            print('adv: {}/{} corrects'.format((t==torch.argmax(class_adv, dim=1)).sum(), t.size(0)))
            print('select diff: {}'.format(torch.norm(select_std-select_adv)))

            # print('std: {}, {}'.format(torch.argmax(class_std, dim=1), select_std))
            # print('adv: {}, {}'.format(torch.argmax(class_adv, dim=1), select_adv))

            raise NotImplementedError

            # evaluator
            # evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())

            # # compute selective loss
            # eval_dict = OrderedDict()
            # eval_dict.update(evaluator())
            # print(eval_dict)


    attacker = PGDAttackVariant()