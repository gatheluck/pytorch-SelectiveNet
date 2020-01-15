import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import OrderedDict

import torch
import torchvision

class Evaluator(object):
    def __init__(self, prediction_out, t, selection_out=None, selection_threshold:float=0.5):
        """
        Args:
            prediction_out (B, #class):
            t (B):  
            selection_out (B, 1)
        """
        assert 0<=selection_threshold<=1.0

        self.prediction_result = prediction_out.argmax(dim=1) # (B)
        self.t = t.detach() # (B)
        if selection_out is not None:
            condition = (selection_out >= selection_threshold)
            self.selection_result = torch.where(condition, torch.ones_like(selection_out), torch.zeros_like(selection_out)).view(-1) # (B)
        else:
            self.selection_result = None

    def __call__(self):
        """
        add 'accuracy (Acc)', 'precision (Pre)', 'recall (Rec)' to metric_dict. 
        if selection_out is not None, 'rejection rate (RR)' and 'precision of rejection (PR)' are added.
        
        Args:
            metric_dict
        """
        eval_dict = OrderedDict()

        if self.selection_result is None:
            # add evaluation for classification
            eval_dict_cls = self._evaluate_multi_classification(self.prediction_result, self.t)
            eval_dict.update(eval_dict_cls)

        else:
            # add evaluation for classification
            eval_dict_cls = self._evaluate_multi_classification_with_rejection(self.prediction_result, self.t, self.selection_result)
            eval_dict.update(eval_dict_cls)
            # add evaluation for rejection 
            eval_dict_rjc = self._evaluate_rejection(self.prediction_result, self.t, self.selection_result)
            eval_dict.update(eval_dict_rjc)

        return eval_dict


    def _evaluate_multi_classification(self, h:torch.tensor, t:torch.tensor):
        """
        evaluate result of multi classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1
            t (B): labels which indicates true label form 0 to #class-1
        Return:
            OrderedDict: accuracy
        """
        assert h.size(0) == t.size(0) > 0
        assert len(h.size()) == len(t.size()) == 1

        t = float(torch.where(h==t, torch.ones_like(h), torch.zeros_like(h)).sum())
        f = float(torch.where(h!=t, torch.ones_like(h), torch.zeros_like(h)).sum())

        # raw accuracy
        acc = float(t/(t+f+1e-12))
        return OrderedDict({'accuracy':acc})

    def _evaluate_multi_classification_with_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of multi classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1
            t (B): labels which indicates true label form 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: 'acc'/'raw acc'
        """
        assert h.size(0) == t.size(0) == r_binary.size(0)> 0
        assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

        # raw accuracy
        eval_dict = self._evaluate_multi_classification(h, t)
        eval_dict['raw accuracy'] = eval_dict['accuracy']
        del eval_dict['accuracy']

        h_rjc = torch.masked_select(h, r_binary.bool())
        t_rjc = torch.masked_select(t, r_binary.bool())

        t = float(torch.where(h_rjc==t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc!=t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        # accuracy
        acc = float(t/(t+f+1e-12))
        eval_dict['accuracy'] = acc

        return eval_dict


    def _evaluate_binary_classification(self, h:torch.tensor, t_binary:torch.tensor):
        """
        evaluate result of binary classification. 

        Args:
            h (B): binary prediction which indicates 'positive:1' and 'negative:0'
            t_binary (B): labels which indicates 'true1:' and 'false:0'
        Return:
            OrderedDict: accuracy, precision, recall
        """
        assert h.size(0) == t_binary.size(0) > 0
        assert len(h.size()) == len(t_binary.size()) == 1

        # conditions (true,false,positive,negative)
        condition_true  = (h==t_binary)
        condition_false = (h!=t_binary)
        condition_pos = (h==torch.ones_like(h))
        condition_neg = (h==torch.zeros_like(h))

        # TP, TN, FP, FN
        true_pos = torch.where(condition_true and condition_pos, torch.ones_like(h), torch.zeros_like(h))
        true_neg = torch.where(condition_true and condition_neg, torch.ones_like(h), torch.zeros_like(h))
        false_pos = torch.where(condition_false and condition_pos, torch.ones_like(h), torch.zeros_like(h))
        false_neg = torch.where(condition_false and condition_neg, torch.ones_like(h), torch.zeros_like(h))

        assert (true_pos + true_neg + false_pos + false_neg)==torch.ones_like(true_pos)

        tp = float(true_pos.sum())
        tn = float(true_neg.sum())
        fp = float(false_pos.sum())
        fn = float(false_neg.sum())

        # accuracy, precision, recall
        acc = float((tp+tn)/(tp+tn+fp+fn+1e-12))
        pre = float(tp/(tp+fp+1e-12))
        rec = float(tp/(tp+fn+1e-12))

        return OrderedDict({'accuracy':acc, 'precision':pre, 'recall':rec})


    def _evaluate_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of binary classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1 
            t (B): labels which indicates true class index from 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: rejection_rate, rejection_precision
        """
        assert h.size(0) == t.size(0) == r_binary.size(0)> 0
        assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

        # conditions (true,false,positive,negative)
        condition_true  = (h==t)
        condition_false = (h!=t)
        
        condition_acc = (r_binary==torch.ones_like(r_binary))
        condition_rjc = (r_binary==torch.zeros_like(r_binary))

        # TP, TN, FP, FN
        ta = float(torch.where(condition_true & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        tr = float(torch.where(condition_true & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fa = float(torch.where(condition_false & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fr = float(torch.where(condition_false & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())

        # accuracy, precision, recall
        rejection_rate = float((tr+fr)/(ta+tr+fa+fr+1e-12))
        rejection_pre  = float(tr/(tr+fr+1e-12))

        return OrderedDict({'rejection rate':rejection_rate, 'rejection precision':rejection_pre}) 

if __name__ == '__main__':
    from selectivenet.vgg_variant import vgg16_variant
    from selectivenet.model import SelectiveNet
    from selectivenet.loss import SelectiveLoss
    from selectivenet.data import DatasetBuilder

    # dataset
    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    test_dataset   = dataset_builder(train=False, normalize=True)
    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, 0.3).cuda()
    model = SelectiveNet(features, 512, dataset_builder.num_classes).cuda()

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # test
    with torch.autograd.no_grad():
        for i, (x,t) in enumerate(test_loader):
            model.eval()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class, out_select, _ = model(x)

            # evaluator
            evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())

            # compute selective loss
            eval_dict = OrderedDict()
            eval_dict.update(evaluator())
            print(eval_dict)
            


            
            

