import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torchvision

from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss

def test_train_selective_loss():
    # dataset
    mean = [0.49139968, 0.48215841, 0.44653091]
    std  = [0.24703223, 0.24348513, 0.26158784]
    #mean = torch.tensor(mean, dtype=torch.float32).cuda()
    #std  = torch.tensor(std, dtype=torch.float32).cuda()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    data_root = os.path.join('/home/gatheluck/Scratch/selectivenet/data', 'cifar10')
    train_dataset = torchvision.datasets.CIFAR10(data_root, True, transform, download=True)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    # model
    features = torchvision.models.resnet18(num_classes=512).cuda()
    model = SelectiveNet(features, 512, 10).cuda()

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # loss
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=0.7)

    for ep in range(100):
        # train
        for i, (x,t) in enumerate(train_loader):
            model.train()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class, out_select, out_aux = model(x)

            # compute selective loss
            loss_dict = {}
            # loss dict includes, 'empirical_risk' / 'emprical_coverage' / 'penulty'
            selective_loss, loss_dict = SelectiveCELoss(out_class, out_select, t)
            loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
            
            # total loss
            loss = selective_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss_dict)
        
        # validation

        # post epoch
        scheduler.step()

if __name__ == '__main__':
    test_train_selective_loss()