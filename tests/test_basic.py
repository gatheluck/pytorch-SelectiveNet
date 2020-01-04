import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from selectivenet.vgg_variant import make_layers, cfgs
from selectivenet.vgg_variant import vgg11_variant, vgg13_variant, vgg16_variant, vgg19_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss

def test_make_layers():
    mode_length = {'A':33, 'B':41, 'D':53, 'E':65}
    x = torch.randn(16,3,32,32).cuda()

    for mode, length in mode_length.items():
        # length test
        layers = make_layers(cfgs[mode], 0.3).cuda()
        assert len(layers)==length
        print('mode: ', mode)
        print(layers)

        # forward test
        out = layers(x)
        assert out.size(0)==16
        assert out.size(1)==512
        assert out.size(2)==1
        assert out.size(3)==1
        print('output shape:', out.shape)

def test_vgg_variant():
    x = torch.randn(16,3,32,32).cuda()

    model = vgg11_variant(32,0.3).cuda()
    out = model(x)
    assert out.size(0)==16
    assert out.size(1)==512
    print(out.shape)

    model = vgg13_variant(32,0.3).cuda()
    out = model(x)
    assert out.size(0)==16
    assert out.size(1)==512
    print(out.shape)

    model = vgg16_variant(32,0.3).cuda()
    out = model(x)
    assert out.size(0)==16
    assert out.size(1)==512
    print(out.shape)

    model = vgg19_variant(32,0.3).cuda()
    out = model(x)
    assert out.size(0)==16
    assert out.size(1)==512
    print(out.shape)

def test_model():
    x = torch.randn(16,3,32,32).cuda()

    features = vgg11_variant(32,0.3).cuda()
    model = SelectiveNet(features, 512, 10).cuda()
    out_class, out_select, out_aux = model(x)
    assert out_class.size(0)==16
    assert out_class.size(1)==10
    assert out_select.size(0)==16
    assert out_select.size(1)==1
    assert out_aux.size(0)==16
    assert out_aux.size(1)==10
    print('out_class', out_class.shape)
    print('out_select', out_select.shape)
    print('out_aux', out_aux.shape)

    features = vgg13_variant(32,0.3).cuda()
    model = SelectiveNet(features, 512, 10).cuda()
    out_class, out_select, out_aux = model(x)
    assert out_class.size(0)==16
    assert out_class.size(1)==10
    assert out_select.size(0)==16
    assert out_select.size(1)==1
    assert out_aux.size(0)==16
    assert out_aux.size(1)==10
    print('out_class', out_class.shape)
    print('out_select', out_select.shape)
    print('out_aux', out_aux.shape)

    features = vgg16_variant(32,0.3).cuda()
    model = SelectiveNet(features, 512, 10).cuda()
    out_class, out_select, out_aux = model(x)
    assert out_class.size(0)==16
    assert out_class.size(1)==10
    assert out_select.size(0)==16
    assert out_select.size(1)==1
    assert out_aux.size(0)==16
    assert out_aux.size(1)==10
    print('out_class', out_class.shape)
    print('out_select', out_select.shape)
    print('out_aux', out_aux.shape)

    features = vgg19_variant(32,0.3).cuda()
    model = SelectiveNet(features, 512, 10).cuda()
    out_class, out_select, out_aux = model(x)
    assert out_class.size(0)==16
    assert out_class.size(1)==10
    assert out_select.size(0)==16
    assert out_select.size(1)==1
    assert out_aux.size(0)==16
    assert out_aux.size(1)==10
    print('out_class', out_class.shape)
    print('out_select', out_select.shape)
    print('out_aux', out_aux.shape)

def test_selective_loss():
    x = torch.randn(16,3,32,32).cuda()
    features = vgg16_variant(32,0.3).cuda()
    model = SelectiveNet(features, 512, 10).cuda()
    out_class, out_select, out_aux = model(x)
    target = torch.randint(0,9,(16,)).cuda()

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss = SelectiveLoss(loss_func, coverage=0.7)(out_class, out_select, out_aux, target)
    assert loss.size(0)==1
    print('loss', loss)

if __name__ == '__main__':
    test_make_layers()
    test_vgg_variant()
    test_model()
    test_selective_loss()