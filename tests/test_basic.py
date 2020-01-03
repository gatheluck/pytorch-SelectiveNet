import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from selectivenet.vgg_variant import make_layers, cfgs
from selectivenet.vgg_variant import vgg11_variant, vgg13_variant, vgg16_variant, vgg19_variant

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


if __name__ == '__main__':
    test_make_layers()
    test_vgg_variant()