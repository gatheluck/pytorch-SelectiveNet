import torch
from torch.nn.modules.loss import _Loss

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, alpha:float=0.5, lm:float=32.0):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B). 
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            alpha: blancing parameter between selective loss and standard.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32. 
        """
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 <= alpha <= 1.0
        assert 0.0 < lm

        self.loss_func = loss_func
        self.coverage = coverage
        self.alpha = alpha
        self.lm = lm

    def forward(self, prediction_out, selection_out, auxiliary_out, target):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
            auxiliary_out:  (B, num_classes)
        """
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean() 

        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True)
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True))**2

        selective_loss = emprical_risk + self.lm*penulty
        standard_loss = self.loss_func(auxiliary_out, target).mean()

        total_loss = self.alpha*selective_loss + (1.0-self.alpha)*standard_loss

        return total_loss