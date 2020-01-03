
import torch

class VggVariant(torch.nn.Module):
    """
    Variant of VGG used in SelectiveNet (Geifman et al., 2019).
    In the paper, variant of VGG16 is used as body block.
    Different points from original VGG are following.
    - use only one FC layer with 512 neuron.
    - add batch normalization.
    - add dropout. (about detail places of dropout, please also refer author's origial TF implementation. https://github.com/geifmany/SelectiveNet)
    """
    def __init__(self, features, dropout_base_prob:float, input_size:int, init_weights=True):
        """
        Args:
            features: feature extraction layer. 
            dropout_base_prob: base probability of an element to be zeroed by dropout.
            input_size: size of input. (feature_size = input_size/32)
            init_weights: initialize weight or not.
        """

        super(VggVariant, self).__init__()
        self.features = features
        self.feature_size = int(input_size/32)
        self.dropout_base_prob = dropout_base_prob
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512*self.feature_size*self.feature_size , 512),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout_base_prob+0.2),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def make_layers(cfg, dropout_base_prob:float):
    """
    Args:
        cfg: config dict which decides layer compositions.
        dropout_base_prob: base probability of an element to be zeroed by dropout.
    """
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        # set v_next to deside dropout position
        if i==len(cfg)-1:
            v_next = None
        else:
            v_next = cfg[i+1]

        # make layers from cfg
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            
            # add dropout
            if v_next==None:
                dropout_prob = dropout_base_prob+0.2
                layers += [torch.nn.Dropout(dropout_prob)]
        else:
            # adjust dropout probability
            if v==64:
                dropout_prob = dropout_base_prob
            else:
                dropout_prob = dropout_base_prob+0.1

            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(v)]
            
            # add dropout
            if v_next != 'M':
                layers += [torch.nn.Dropout(dropout_prob)]

            in_channels = v
    return torch.nn.Sequential(*layers)

# A:VGG11, B:VGG13, D:VGG16, E:VGG19
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg_variant(cfg, dropout_base_prob, **kwargs):
    features = make_layers(cfg=cfgs[cfg], dropout_base_prob=dropout_base_prob)
    model = VggVariant(features, dropout_base_prob, **kwargs)
    return model

def vgg11_variant(input_size, dropout_base_prob):
    return _vgg_variant('A', dropout_base_prob=dropout_base_prob, input_size=input_size)

def vgg13_variant(input_size, dropout_base_prob):
    return _vgg_variant('B', dropout_base_prob=dropout_base_prob, input_size=input_size)

def vgg16_variant(input_size, dropout_base_prob):
    return _vgg_variant('D', dropout_base_prob=dropout_base_prob, input_size=input_size)

def vgg19_variant(input_size, dropout_base_prob):
    return _vgg_variant('E', dropout_base_prob=dropout_base_prob, input_size=input_size)